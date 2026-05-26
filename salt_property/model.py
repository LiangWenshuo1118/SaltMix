from __future__ import annotations

import torch
from torch import nn

from .config_utils import normalize_bool, normalize_rk_order, normalize_temperature_law


class SetTransformerPool(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.self_ff_norm = nn.LayerNorm(dim)
        self.seed = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.seed)
        self.seed_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.seed_norm = nn.LayerNorm(dim)
        self.seed_ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.seed_ff_norm = nn.LayerNorm(dim)

    def forward(self, values: torch.Tensor, component_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = component_mask <= 0
        safe_key_padding_mask = key_padding_mask.clone()
        all_padding = safe_key_padding_mask.all(dim=1)
        if torch.any(all_padding):
            safe_key_padding_mask[all_padding, 0] = False

        attended, _ = self.self_attention(
            values,
            values,
            values,
            key_padding_mask=safe_key_padding_mask,
            need_weights=False,
        )
        values = self.self_norm(values + attended)
        values = self.self_ff_norm(values + self.self_ff(values))
        values = values.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        seeds = self.seed.expand(values.shape[0], -1, -1)
        pooled, _ = self.seed_attention(
            seeds,
            values,
            values,
            key_padding_mask=safe_key_padding_mask,
            need_weights=False,
        )
        pooled = self.seed_norm(seeds + pooled)
        pooled = self.seed_ff_norm(pooled + self.seed_ff(pooled))
        return pooled.squeeze(1)


class PropertyRKModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        encoder_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        subnet_dim: int = 128,
        rk_order: int | None = 2,
        use_phy: bool = True,
        use_arrhenius: bool = False,
        use_linear: bool = False,
        temperature_law: str | None = None,
        max_components: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_phy = normalize_bool(use_phy, field_name="use_phy")
        if temperature_law is None:
            arrhenius_enabled = normalize_bool(use_arrhenius, field_name="use_arrhenius")
            linear_enabled = normalize_bool(use_linear, field_name="use_linear")
            if arrhenius_enabled and linear_enabled:
                raise ValueError("use_arrhenius and use_linear cannot both be true")
            if arrhenius_enabled:
                temperature_law = "arrhenius"
            elif linear_enabled:
                temperature_law = "linear"
            else:
                temperature_law = "none"
        self.temperature_law = normalize_temperature_law(temperature_law)
        self.use_arrhenius = self.temperature_law == "arrhenius"
        self.use_linear = self.temperature_law == "linear"
        if self.temperature_law != "none" and not self.use_phy:
            raise ValueError("use_arrhenius/use_linear requires use_phy: true")
        self.max_components = max_components
        self.rk_order = None if (not self.use_phy or self.temperature_law != "none" or rk_order is None) else normalize_rk_order(rk_order)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        projection_input_dim = embed_dim + 2
        self.component_projection = nn.Linear(projection_input_dim, encoder_dim)
        self.subnet = nn.Sequential(
            nn.Linear(encoder_dim, subnet_dim),
            nn.ReLU(),
            nn.Linear(subnet_dim, subnet_dim),
            nn.ReLU(),
        )
        self.set_pool = None if self.use_phy else SetTransformerPool(
            subnet_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        if self.temperature_law != "none":
            self.pure_head = None
            self.pair_head = None
            self.direct_head = None
            law_pair_head = nn.Sequential(
                nn.Linear(subnet_dim * 2, subnet_dim),
                nn.ReLU(),
                nn.Linear(subnet_dim, subnet_dim),
                nn.ReLU(),
            )
            law_head = nn.Sequential(
                nn.Linear(subnet_dim, subnet_dim),
                nn.ReLU(),
                nn.Linear(subnet_dim, 2),
            )
            if self.temperature_law == "arrhenius":
                self.arrhenius_pair_head = law_pair_head
                self.arrhenius_head = law_head
                self.linear_pair_head = None
                self.linear_head = None
            else:
                self.arrhenius_pair_head = None
                self.arrhenius_head = None
                self.linear_pair_head = law_pair_head
                self.linear_head = law_head
        elif self.use_phy:
            self.pure_head = nn.Linear(subnet_dim, 1)
            if self.rk_order is None:
                self.pair_head = None
            else:
                self.pair_head = nn.Sequential(
                    nn.Linear(subnet_dim * 2, subnet_dim),
                    nn.ReLU(),
                    nn.Linear(subnet_dim, self.rk_order + 1),
                )
            self.direct_head = None
            self.arrhenius_pair_head = None
            self.arrhenius_head = None
            self.linear_pair_head = None
            self.linear_head = None
        else:
            self.pure_head = None
            self.pair_head = None
            self.direct_head = nn.Sequential(
                nn.Linear(subnet_dim, subnet_dim),
                nn.ReLU(),
                nn.Linear(subnet_dim, subnet_dim),
                nn.ReLU(),
                nn.Linear(subnet_dim, 1),
            )
            self.arrhenius_pair_head = None
            self.arrhenius_head = None
            self.linear_pair_head = None
            self.linear_head = None

    def encode_components(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_mask = token_ids != 0
        embeddings = self.embedding(token_ids)
        batch, num_components, num_tokens, dim = embeddings.shape
        flat_embeddings = embeddings.view(batch * num_components, num_tokens, dim)
        flat_mask = token_mask.view(batch * num_components, num_tokens)
        safe_mask = flat_mask.clone()
        all_padding = ~safe_mask.any(dim=1)
        safe_mask[all_padding, 0] = True
        encoded = self.encoder(flat_embeddings, src_key_padding_mask=~safe_mask)
        pooled = (encoded * safe_mask.unsqueeze(-1)).sum(dim=1) / safe_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return pooled.view(batch, num_components, dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        mole_fractions: torch.Tensor,
        temperatures: torch.Tensor,
        component_mask: torch.Tensor,
        raw_temperatures: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded_components = self.encode_components(token_ids)
        temperature_features = temperatures.unsqueeze(1).expand(-1, mole_fractions.shape[1]).unsqueeze(-1)
        component_inputs = torch.cat(
            [encoded_components, mole_fractions.unsqueeze(-1), temperature_features],
            dim=-1,
        )
        hidden = self.component_projection(component_inputs)
        latent = self.subnet(hidden)
        if self.temperature_law != "none":
            if self.temperature_law == "arrhenius":
                pair_head = self.arrhenius_pair_head
                law_head = self.arrhenius_head
            else:
                pair_head = self.linear_pair_head
                law_head = self.linear_head
            assert pair_head is not None
            assert law_head is not None
            mixture_features = (latent * mole_fractions.unsqueeze(-1) * component_mask.unsqueeze(-1)).sum(dim=1)
            max_components = mole_fractions.shape[1]
            for i in range(max_components):
                for j in range(i + 1, max_components):
                    valid_pair = component_mask[:, i] * component_mask[:, j]
                    if torch.count_nonzero(valid_pair) == 0:
                        continue
                    pair_ij = torch.cat([latent[:, i], latent[:, j]], dim=-1)
                    pair_ji = torch.cat([latent[:, j], latent[:, i]], dim=-1)
                    pair_features = 0.5 * (
                        pair_head(pair_ij)
                        + pair_head(pair_ji)
                    )
                    pair_weight = valid_pair * mole_fractions[:, i] * mole_fractions[:, j]
                    mixture_features = mixture_features + pair_weight.unsqueeze(-1) * pair_features
            params = law_head(mixture_features)
            intercept = params[:, 0]
            slope = params[:, 1]
            law_temperatures = temperatures if raw_temperatures is None else raw_temperatures
            if self.temperature_law == "arrhenius":
                return intercept + slope / law_temperatures.clamp_min(1e-6)
            return intercept + slope * law_temperatures

        if not self.use_phy:
            assert self.set_pool is not None
            direct_inputs = self.set_pool(latent, component_mask)
            assert self.direct_head is not None
            return self.direct_head(direct_inputs).squeeze(-1)

        assert self.pure_head is not None
        pure_values = self.pure_head(latent).squeeze(-1)
        mixture = (pure_values * mole_fractions * component_mask).sum(dim=1)
        if self.rk_order is None:
            return mixture

        batch_size, max_components = mole_fractions.shape
        exchange_signs = torch.ones(self.rk_order + 1, device=latent.device, dtype=latent.dtype)
        exchange_signs[1::2] = -1
        for i in range(max_components):
            for j in range(i + 1, max_components):
                valid_pair = component_mask[:, i] * component_mask[:, j]
                if torch.count_nonzero(valid_pair) == 0:
                    continue
                pair_ij = torch.cat([latent[:, i], latent[:, j]], dim=-1)
                pair_ji = torch.cat([latent[:, j], latent[:, i]], dim=-1)
                assert self.pair_head is not None
                coeffs_ij = self.pair_head(pair_ij)
                coeffs_ji = self.pair_head(pair_ji)
                coeffs = 0.5 * (coeffs_ij + exchange_signs * coeffs_ji)
                delta = mole_fractions[:, i] - mole_fractions[:, j]
                rk_sum = torch.zeros(batch_size, device=token_ids.device)
                for k in range(self.rk_order + 1):
                    rk_sum = rk_sum + coeffs[:, k] * delta.pow(k)
                mixture = mixture + valid_pair * mole_fractions[:, i] * mole_fractions[:, j] * rk_sum
        return mixture
