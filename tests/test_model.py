from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest


class ModelSmokeTests(unittest.TestCase):
    def test_resolve_model_rk_order_uses_explicit_use_rk_flag(self):
        from salt_mix.config_utils import (
            normalize_rk_order,
            resolve_model_rk_order,
            resolve_model_temperature_law,
            resolve_model_use_arrhenius,
            resolve_model_use_phy,
        )

        self.assertEqual(normalize_rk_order("2"), 2)
        self.assertEqual(normalize_rk_order(4), 4)
        self.assertTrue(resolve_model_use_phy({}))
        self.assertTrue(resolve_model_use_phy({"use_phy": "true"}))
        self.assertFalse(resolve_model_use_phy({"use_phy": False}))
        self.assertFalse(resolve_model_use_arrhenius({}))
        self.assertTrue(resolve_model_use_arrhenius({"use_arrhenius": True, "use_phy": True}))
        self.assertEqual(resolve_model_temperature_law({}), "none")
        self.assertEqual(resolve_model_temperature_law({"use_arrhenius": True, "use_phy": True}), "arrhenius")
        self.assertEqual(resolve_model_temperature_law({"use_linear": True, "use_phy": True}), "linear")
        self.assertIsNone(resolve_model_rk_order({"use_phy": False, "use_rk": True, "rk_order": 2}))
        self.assertIsNone(resolve_model_rk_order({"use_rk": False, "rk_order": 2}))
        self.assertIsNone(resolve_model_rk_order({"use_rk": "false", "rk_order": 2}))
        self.assertIsNone(resolve_model_rk_order({"use_arrhenius": True, "use_rk": False, "rk_order": 2}))
        self.assertIsNone(resolve_model_rk_order({"use_linear": True, "use_rk": False, "rk_order": 2}))
        self.assertEqual(resolve_model_rk_order({"use_rk": True, "rk_order": 2}), 2)
        self.assertEqual(resolve_model_rk_order({"rk_order": 4}), 4)
        with self.assertRaises(ValueError):
            resolve_model_use_arrhenius({"use_arrhenius": True, "use_phy": False})
        with self.assertRaises(ValueError):
            resolve_model_rk_order({"use_arrhenius": True, "use_rk": True, "rk_order": 2})
        with self.assertRaises(ValueError):
            resolve_model_temperature_law({"use_arrhenius": True, "use_linear": True, "use_phy": True})
        with self.assertRaises(ValueError):
            resolve_model_rk_order({"use_linear": True, "use_rk": True, "rk_order": 2})
        with self.assertRaises(ValueError):
            normalize_rk_order(None)
        with self.assertRaises(ValueError):
            normalize_rk_order("")
        with self.assertRaises(ValueError):
            normalize_rk_order("none")
        with self.assertRaises(ValueError):
            normalize_rk_order("null")

    def test_import_or_skip(self):
        try:
            import torch
            from salt_density.model import DensityRKModel
            from salt_mix.model import PropertyRKModel
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        model = DensityRKModel(vocab_size=32, rk_order=2)
        token_ids = torch.randint(0, 31, (2, 3, 10))
        mole_fractions = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float32)
        temperatures = torch.tensor([0.0, 1.0], dtype=torch.float32)
        component_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        outputs = model(token_ids, mole_fractions, temperatures, component_mask)
        self.assertEqual(tuple(outputs.shape), (2,))

        generic_model = PropertyRKModel(vocab_size=32, rk_order=2)
        self.assertEqual(generic_model.component_projection.in_features, 130)
        generic_outputs = generic_model(token_ids, mole_fractions, temperatures, component_mask)
        self.assertEqual(tuple(generic_outputs.shape), (2,))

    def test_property_model_rk_order_none_skips_pair_head(self):
        try:
            import torch
            from salt_mix.model import PropertyRKModel
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        model = PropertyRKModel(
            vocab_size=32,
            embed_dim=8,
            encoder_dim=8,
            num_heads=1,
            num_layers=1,
            ff_dim=16,
            subnet_dim=8,
            rk_order=None,
            dropout=0.0,
        )
        token_ids = torch.randint(0, 31, (2, 3, 6))
        mole_fractions = torch.tensor([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float32)
        temperatures = torch.tensor([0.0, 1.0], dtype=torch.float32)
        component_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)

        outputs = model(token_ids, mole_fractions, temperatures, component_mask)

        self.assertIsNone(model.rk_order)
        self.assertIsNone(model.pair_head)
        self.assertEqual(model.component_projection.in_features, 10)
        self.assertFalse(any(key.startswith("pair_head.") for key in model.state_dict()))
        self.assertEqual(tuple(outputs.shape), (2,))

    def test_property_model_rk_terms_are_permutation_invariant(self):
        try:
            import torch
            from salt_mix.model import PropertyRKModel
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        torch.manual_seed(7)
        model = PropertyRKModel(
            vocab_size=32,
            embed_dim=8,
            encoder_dim=8,
            num_heads=1,
            num_layers=1,
            ff_dim=16,
            subnet_dim=8,
            rk_order=3,
            dropout=0.0,
        )
        model.eval()
        token_ids = torch.tensor([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]]], dtype=torch.long)
        mole_fractions = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
        temperatures = torch.tensor([0.4], dtype=torch.float32)
        component_mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        permutation = torch.tensor([2, 0, 1])

        reference = model(token_ids, mole_fractions, temperatures, component_mask)
        permuted = model(
            token_ids[:, permutation],
            mole_fractions[:, permutation],
            temperatures,
            component_mask[:, permutation],
        )

        self.assertTrue(torch.allclose(reference, permuted, atol=1e-6))

    def test_property_model_without_physics_uses_pooled_direct_head(self):
        try:
            import torch
            from salt_mix.model import PropertyRKModel
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        model = PropertyRKModel(
            vocab_size=32,
            embed_dim=8,
            encoder_dim=8,
            num_heads=1,
            num_layers=1,
            ff_dim=16,
            subnet_dim=8,
            rk_order=2,
            use_phy=False,
            max_components=3,
            dropout=0.0,
        )
        model.eval()
        token_ids = torch.tensor(
            [
                [[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0]],
                [[7, 8, 9, 0], [10, 11, 12, 0], [13, 14, 15, 0]],
            ],
            dtype=torch.long,
        )
        mole_fractions = torch.tensor([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float32)
        temperatures = torch.tensor([0.0, 1.0], dtype=torch.float32)
        component_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        permutation = torch.tensor([1, 0, 2])

        outputs = model(token_ids, mole_fractions, temperatures, component_mask)
        permuted_outputs = model(
            token_ids[:, permutation],
            mole_fractions[:, permutation],
            temperatures,
            component_mask[:, permutation],
        )

        self.assertFalse(model.use_phy)
        self.assertIsNone(model.rk_order)
        self.assertIsNotNone(model.component_projection)
        self.assertEqual(model.component_projection.in_features, 10)
        self.assertIsNotNone(model.set_pool)
        self.assertIsNotNone(model.subnet)
        self.assertIsNone(model.pure_head)
        self.assertIsNone(model.pair_head)
        self.assertIsNotNone(model.direct_head)
        self.assertFalse(any(key.startswith("pure_head.") for key in model.state_dict()))
        self.assertFalse(any(key.startswith("pair_head.") for key in model.state_dict()))
        self.assertEqual(tuple(outputs.shape), (2,))
        self.assertTrue(torch.allclose(outputs, permuted_outputs, atol=1e-6))

    def test_set_transformer_pool_is_permutation_invariant(self):
        try:
            import torch
            from salt_mix.model import SetTransformerPool
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        torch.manual_seed(11)
        pool = SetTransformerPool(dim=4, num_heads=1, dropout=0.0)
        pool.eval()
        values = torch.tensor([[[1.0, 0.0, 0.5, 0.2], [0.0, 1.0, 0.4, 0.3], [0.0, 0.0, 0.0, 0.0]]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        permutation = torch.tensor([1, 0, 2])

        reference = pool(values, mask)
        permuted = pool(values[:, permutation], mask[:, permutation])

        self.assertTrue(torch.allclose(reference, permuted, atol=1e-6))

    def test_property_model_arrhenius_route_uses_raw_kelvin_temperature(self):
        try:
            import torch
            from salt_mix.model import PropertyRKModel
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        class FixedArrheniusHead(torch.nn.Module):
            def forward(self, values):
                params = torch.zeros(values.shape[0], 2, dtype=values.dtype, device=values.device)
                params[:, 0] = 1.5
                params[:, 1] = 200.0
                return params

        class ZeroProjection(torch.nn.Module):
            def forward(self, values):
                return torch.zeros(values.shape[0], values.shape[1], 2, dtype=values.dtype, device=values.device)

        model = PropertyRKModel(
            vocab_size=8,
            embed_dim=1,
            encoder_dim=2,
            num_heads=1,
            num_layers=1,
            ff_dim=4,
            subnet_dim=2,
            use_arrhenius=True,
            rk_order=None,
            dropout=0.0,
        )
        model.encode_components = lambda token_ids: torch.zeros(  # type: ignore[method-assign]
            token_ids.shape[0],
            token_ids.shape[1],
            1,
            dtype=torch.float32,
            device=token_ids.device,
        )
        model.component_projection = ZeroProjection()
        model.subnet = torch.nn.Identity()
        model.arrhenius_head = FixedArrheniusHead()

        token_ids = torch.ones(2, 3, 2, dtype=torch.long)
        mole_fractions = torch.tensor([[1.0, 0.0, 0.0], [0.4, 0.6, 0.0]], dtype=torch.float32)
        temperatures = torch.tensor([1000.0, 500.0], dtype=torch.float32)
        component_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32)

        outputs = model(token_ids, mole_fractions, temperatures, component_mask)

        expected = torch.tensor([1.7, 1.9], dtype=torch.float32)
        self.assertTrue(torch.allclose(outputs, expected, atol=1e-6))

        standardized_temperatures = torch.tensor([0.0, 1.0], dtype=torch.float32)
        outputs_with_raw = model(
            token_ids,
            mole_fractions,
            standardized_temperatures,
            component_mask,
            raw_temperatures=temperatures,
        )

        self.assertTrue(torch.allclose(outputs_with_raw, expected, atol=1e-6))

    def test_property_model_arrhenius_route_builds_symmetric_pair_mixture(self):
        try:
            import torch
            from salt_mix.model import PropertyRKModel
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        class SelectEncoded(torch.nn.Module):
            def forward(self, values):
                return values[:, :, :2]

        class FixedPairHead(torch.nn.Module):
            def forward(self, values):
                left = values[:, :2]
                right = values[:, 2:]
                return torch.stack(
                    [
                        left[:, 0] + 2.0 * right[:, 0],
                        left[:, 1] + 3.0 * right[:, 1],
                    ],
                    dim=1,
                )

        class IdentityArrheniusHead(torch.nn.Module):
            def forward(self, values):
                return values[:, :2]

        model = PropertyRKModel(
            vocab_size=8,
            embed_dim=2,
            encoder_dim=2,
            num_heads=1,
            num_layers=1,
            ff_dim=4,
            subnet_dim=2,
            use_arrhenius=True,
            rk_order=None,
            dropout=0.0,
        )
        model.encode_components = lambda token_ids: torch.tensor(  # type: ignore[method-assign]
            [[[2.0, 0.0], [0.0, 4.0]]],
            dtype=torch.float32,
            device=token_ids.device,
        )
        model.component_projection = SelectEncoded()
        model.subnet = torch.nn.Identity()
        model.arrhenius_pair_head = FixedPairHead()
        model.arrhenius_head = IdentityArrheniusHead()

        token_ids = torch.ones(1, 2, 2, dtype=torch.long)
        mole_fractions = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
        temperatures = torch.tensor([2.0], dtype=torch.float32)
        component_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

        output = model(token_ids, mole_fractions, temperatures, component_mask)

        expected = torch.tensor([3.3125], dtype=torch.float32)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_property_model_arrhenius_route_is_permutation_invariant(self):
        try:
            import torch
            from salt_mix.model import PropertyRKModel
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        torch.manual_seed(17)
        model = PropertyRKModel(
            vocab_size=32,
            embed_dim=8,
            encoder_dim=8,
            num_heads=1,
            num_layers=1,
            ff_dim=16,
            subnet_dim=8,
            use_arrhenius=True,
            rk_order=None,
            dropout=0.0,
        )
        model.eval()
        token_ids = torch.tensor([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]]], dtype=torch.long)
        mole_fractions = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
        temperatures = torch.tensor([0.4], dtype=torch.float32)
        raw_temperatures = torch.tensor([900.0], dtype=torch.float32)
        component_mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        permutation = torch.tensor([2, 0, 1])

        reference = model(token_ids, mole_fractions, temperatures, component_mask, raw_temperatures)
        permuted = model(
            token_ids[:, permutation],
            mole_fractions[:, permutation],
            temperatures,
            component_mask[:, permutation],
            raw_temperatures,
        )

        self.assertIsNone(model.set_pool)
        self.assertIsNotNone(model.arrhenius_pair_head)
        self.assertTrue(torch.allclose(reference, permuted, atol=1e-6))

    def test_property_model_linear_route_uses_raw_kelvin_temperature(self):
        try:
            import torch
            from salt_mix.model import PropertyRKModel
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        class FixedLinearHead(torch.nn.Module):
            def forward(self, values):
                params = torch.zeros(values.shape[0], 2, dtype=values.dtype, device=values.device)
                params[:, 0] = 3.0
                params[:, 1] = -0.01
                return params

        class ZeroProjection(torch.nn.Module):
            def forward(self, values):
                return torch.zeros(values.shape[0], values.shape[1], 2, dtype=values.dtype, device=values.device)

        model = PropertyRKModel(
            vocab_size=8,
            embed_dim=1,
            encoder_dim=2,
            num_heads=1,
            num_layers=1,
            ff_dim=4,
            subnet_dim=2,
            use_linear=True,
            rk_order=None,
            dropout=0.0,
        )
        model.encode_components = lambda token_ids: torch.zeros(  # type: ignore[method-assign]
            token_ids.shape[0],
            token_ids.shape[1],
            1,
            dtype=torch.float32,
            device=token_ids.device,
        )
        model.component_projection = ZeroProjection()
        model.subnet = torch.nn.Identity()
        model.linear_head = FixedLinearHead()

        token_ids = torch.ones(2, 3, 2, dtype=torch.long)
        mole_fractions = torch.tensor([[1.0, 0.0, 0.0], [0.4, 0.6, 0.0]], dtype=torch.float32)
        temperatures = torch.tensor([100.0, 200.0], dtype=torch.float32)
        component_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32)

        outputs = model(token_ids, mole_fractions, temperatures, component_mask)

        expected = torch.tensor([2.0, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(outputs, expected, atol=1e-6))

        standardized_temperatures = torch.tensor([0.0, 1.0], dtype=torch.float32)
        outputs_with_raw = model(
            token_ids,
            mole_fractions,
            standardized_temperatures,
            component_mask,
            raw_temperatures=temperatures,
        )

        self.assertTrue(torch.allclose(outputs_with_raw, expected, atol=1e-6))

    def test_property_model_linear_route_is_permutation_invariant(self):
        try:
            import torch
            from salt_mix.model import PropertyRKModel
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        torch.manual_seed(19)
        model = PropertyRKModel(
            vocab_size=32,
            embed_dim=8,
            encoder_dim=8,
            num_heads=1,
            num_layers=1,
            ff_dim=16,
            subnet_dim=8,
            use_linear=True,
            rk_order=None,
            dropout=0.0,
        )
        model.eval()
        token_ids = torch.tensor([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]]], dtype=torch.long)
        mole_fractions = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
        temperatures = torch.tensor([0.4], dtype=torch.float32)
        raw_temperatures = torch.tensor([900.0], dtype=torch.float32)
        component_mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        permutation = torch.tensor([2, 0, 1])

        reference = model(token_ids, mole_fractions, temperatures, component_mask, raw_temperatures)
        permuted = model(
            token_ids[:, permutation],
            mole_fractions[:, permutation],
            temperatures,
            component_mask[:, permutation],
            raw_temperatures,
        )

        self.assertTrue(model.use_linear)
        self.assertIsNone(model.rk_order)
        self.assertIsNotNone(model.linear_pair_head)
        self.assertTrue(torch.allclose(reference, permuted, atol=1e-6))

    def test_viscosity_eval_reports_exp_of_log_predictions(self):
        try:
            import torch
            from salt_mix.train import build_dataloader, evaluate_model
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        class EchoTemperatureModel(torch.nn.Module):
            def forward(self, token_ids, mole_fractions, temperatures, component_mask):
                return temperatures

        encoded_samples = [
            {
                "record_id": "V1",
                "system": "AgBr",
                "token_ids": [[1, 2], [0, 0], [0, 0]],
                "component_mask": [1, 0, 0],
                "mole_fractions": [1.0, 0.0, 0.0],
                "temperature": 0.0,
                "target_value": 1.0,
                "training_target": 0.0,
                "target_log_value": 0.0,
            },
            {
                "record_id": "V2",
                "system": "AgBr-KBr",
                "token_ids": [[1, 2], [3, 4], [0, 0]],
                "component_mask": [1, 1, 0],
                "mole_fractions": [0.5, 0.5, 0.0],
                "temperature": math.log(2.0),
                "target_value": 2.0,
                "training_target": math.log(2.0),
                "target_log_value": math.log(2.0),
            },
        ]
        dataloader = build_dataloader(encoded_samples, batch_size=2, shuffle=False)
        first_batch = next(iter(dataloader))
        self.assertTrue(torch.allclose(first_batch["temperature"], torch.tensor([0.0, math.log(2.0)])))
        metrics, predictions = evaluate_model(EchoTemperatureModel(), dataloader, torch.device("cpu"), "viscosity")

        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["rmse_log"], 0.0)
        self.assertAlmostEqual(predictions[0]["prediction_value"], math.exp(predictions[0]["prediction_log_value"]))
        self.assertAlmostEqual(predictions[1]["prediction_value"], math.exp(predictions[1]["prediction_log_value"]))

    def test_density_eval_uses_unified_output_fields(self):
        try:
            import torch
            from salt_mix.train import build_dataloader, evaluate_model
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        class EchoTemperatureModel(torch.nn.Module):
            def forward(self, token_ids, mole_fractions, temperatures, component_mask):
                return temperatures

        encoded_samples = [
            {
                "record_id": "D1",
                "system": "NaCl",
                "token_ids": [[1, 2], [0, 0], [0, 0]],
                "component_mask": [1, 0, 0],
                "mole_fractions": [1.0, 0.0, 0.0],
                "temperature": 1.23,
                "target_value": 1.23,
                "training_target": 1.23,
            }
        ]
        dataloader = build_dataloader(encoded_samples, batch_size=1, shuffle=False)
        metrics, predictions = evaluate_model(EchoTemperatureModel(), dataloader, torch.device("cpu"), "density")

        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertIsNone(metrics["rmse_log"])
        self.assertEqual(predictions[0]["prediction_value"], 1.23)
        self.assertEqual(predictions[0]["target_value"], 1.23)
        self.assertIsNone(predictions[0]["prediction_log_value"])
        self.assertIsNone(predictions[0]["target_log_value"])

    def test_electrical_conductivity_eval_reports_exp_of_log_predictions(self):
        try:
            import torch
            from salt_mix.train import build_dataloader, evaluate_model
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        class EchoTemperatureModel(torch.nn.Module):
            def forward(self, token_ids, mole_fractions, temperatures, component_mask):
                return temperatures

        encoded_samples = [
            {
                "record_id": "E1",
                "system": "AgBr",
                "token_ids": [[1, 2], [0, 0], [0, 0]],
                "component_mask": [1, 0, 0],
                "mole_fractions": [1.0, 0.0, 0.0],
                "temperature": 0.0,
                "target_value": 1.0,
                "training_target": 0.0,
                "target_log_value": 0.0,
            },
            {
                "record_id": "E2",
                "system": "AgBr-KBr",
                "token_ids": [[1, 2], [3, 4], [0, 0]],
                "component_mask": [1, 1, 0],
                "mole_fractions": [0.5, 0.5, 0.0],
                "temperature": math.log(3.0),
                "target_value": 3.0,
                "training_target": math.log(3.0),
                "target_log_value": math.log(3.0),
            },
        ]
        dataloader = build_dataloader(encoded_samples, batch_size=2, shuffle=False)
        metrics, predictions = evaluate_model(
            EchoTemperatureModel(),
            dataloader,
            torch.device("cpu"),
            "electrical_conductivity",
        )

        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["rmse_log"], 0.0)
        self.assertAlmostEqual(predictions[0]["prediction_value"], math.exp(predictions[0]["prediction_log_value"]))
        self.assertAlmostEqual(predictions[1]["prediction_value"], math.exp(predictions[1]["prediction_log_value"]))

    def test_surface_tension_eval_uses_unified_output_fields(self):
        try:
            import torch
            from salt_mix.train import build_dataloader, evaluate_model
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        class EchoTemperatureModel(torch.nn.Module):
            def forward(self, token_ids, mole_fractions, temperatures, component_mask):
                return temperatures

        encoded_samples = [
            {
                "record_id": "S1",
                "system": "AgBr",
                "token_ids": [[1, 2], [0, 0], [0, 0]],
                "component_mask": [1, 0, 0],
                "mole_fractions": [1.0, 0.0, 0.0],
                "temperature": 1.23,
                "target_value": 1.23,
                "training_target": 1.23,
            }
        ]
        dataloader = build_dataloader(encoded_samples, batch_size=1, shuffle=False)
        metrics, predictions = evaluate_model(EchoTemperatureModel(), dataloader, torch.device("cpu"), "surface_tension")

        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertIsNone(metrics["rmse_log"])
        self.assertEqual(predictions[0]["prediction_value"], 1.23)
        self.assertEqual(predictions[0]["target_value"], 1.23)
        self.assertIsNone(predictions[0]["prediction_log_value"])
        self.assertIsNone(predictions[0]["target_log_value"])

    def test_write_epoch_history_outputs_density_csv(self):
        from salt_mix.train import write_epoch_history

        history = [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "val_loss": 0.45,
                "val_rmse": 10.0,
                "val_mae": 8.0,
                "val_pearson_r": 0.1,
                "val_rmse_log": 0.4,
                "val_mae_log": 0.3,
                "val_pearson_r_log": 0.2,
            },
            {
                "epoch": 2,
                "train_loss": 0.25,
                "val_loss": 0.2,
                "val_rmse": 8.0,
                "val_mae": 6.0,
                "val_pearson_r": 0.4,
                "val_rmse_log": 0.3,
                "val_mae_log": 0.2,
                "val_pearson_r_log": 0.5,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_epoch_history(output_dir, history, "density")

            csv_text = (output_dir / "epoch_metrics.csv").read_text(encoding="utf-8")

        self.assertIn("epoch,train_loss,val_loss,val_rmse,val_mae", csv_text)
        self.assertIn("1,0.5,0.45,10.0,8.0", csv_text)
        self.assertIn("2,0.25,0.2,8.0,6.0", csv_text)
        self.assertNotIn("pearson", csv_text)
        self.assertNotIn("rmse_log", csv_text)

    def test_write_epoch_history_outputs_log_metric_csv(self):
        from salt_mix.train import write_epoch_history

        history = [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "val_loss": 0.45,
                "val_rmse": 10.0,
                "val_mae": 8.0,
                "val_pearson_r": 0.1,
                "val_rmse_log": 0.4,
                "val_mae_log": 0.3,
                "val_pearson_r_log": 0.2,
            },
            {
                "epoch": 2,
                "train_loss": 0.25,
                "val_loss": 0.2,
                "val_rmse": 8.0,
                "val_mae": 6.0,
                "val_pearson_r": 0.4,
                "val_rmse_log": 0.3,
                "val_mae_log": 0.2,
                "val_pearson_r_log": 0.5,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_epoch_history(output_dir, history, "viscosity")

            csv_text = (output_dir / "epoch_metrics.csv").read_text(encoding="utf-8")

        self.assertIn("epoch,train_loss,val_loss,val_rmse_log,val_mae_log", csv_text)
        self.assertIn("1,0.5,0.45,0.4,0.3", csv_text)
        self.assertIn("2,0.25,0.2,0.3,0.2", csv_text)
        self.assertNotIn("pearson", csv_text)
        self.assertNotIn("val_rmse,val_mae", csv_text)

    def test_compute_metrics_reports_pearson_r(self):
        from salt_mix.train import compute_metrics

        positive = compute_metrics([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        negative = compute_metrics([1.0, 2.0, 3.0], [30.0, 20.0, 10.0])

        self.assertAlmostEqual(positive["pearson_r"], 1.0)
        self.assertAlmostEqual(negative["pearson_r"], -1.0)

    def test_build_loss_fn_log_cosh_or_skip(self):
        try:
            import torch
            from salt_mix.train import build_loss_fn
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        loss_fn = build_loss_fn("log_cosh")
        zero_loss = loss_fn(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
        nonzero_loss = loss_fn(torch.tensor([2.0]), torch.tensor([0.0]))

        self.assertAlmostEqual(float(zero_loss), 0.0, places=6)
        self.assertGreater(float(nonzero_loss), 0.0)

    def test_update_ema_state_or_skip(self):
        try:
            import torch
            from salt_mix.train import clone_state_dict, update_ema_state
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.0)
        ema_state = clone_state_dict(model.state_dict())
        with torch.no_grad():
            model.weight.fill_(4.0)

        update_ema_state(ema_state, model, 0.98)

        self.assertAlmostEqual(float(ema_state["weight"][0][0]), 2.04, places=6)

    def test_evaluate_with_state_restores_model_or_skip(self):
        try:
            import torch
            from salt_mix.train import build_dataloader, build_loss_fn, clone_state_dict, evaluate_with_state
        except ModuleNotFoundError:
            self.skipTest("torch is not available locally")
            return

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, token_ids, mole_fractions, temperatures, component_mask):
                return temperatures * self.scale

        encoded_samples = [
            {
                "record_id": "D1",
                "system": "NaCl",
                "token_ids": [[1, 2], [0, 0], [0, 0]],
                "component_mask": [1, 0, 0],
                "mole_fractions": [1.0, 0.0, 0.0],
                "temperature": 2.0,
                "target_value": 4.0,
                "training_target": 4.0,
            }
        ]
        model = TinyModel()
        dataloader = build_dataloader(encoded_samples, batch_size=1, shuffle=False)
        target_state = clone_state_dict(model.state_dict())
        target_state["scale"].fill_(2.0)

        val_loss, metrics, _ = evaluate_with_state(
            model,
            target_state,
            dataloader,
            torch.device("cpu"),
            "density",
            build_loss_fn("mse"),
        )

        self.assertAlmostEqual(val_loss, 0.0, places=6)
        self.assertAlmostEqual(metrics["rmse"], 0.0, places=6)
        self.assertAlmostEqual(float(model.scale.detach().cpu()), 1.0, places=6)

    def test_csv_safe_rows_converts_none_to_nan(self):
        from salt_mix.train import csv_safe_rows

        rows = [
            {"prediction_value": 1.0, "prediction_log_value": None, "target_log_value": None},
        ]

        safe_rows = csv_safe_rows(rows)

        self.assertEqual(safe_rows[0]["prediction_log_value"], "nan")
        self.assertEqual(safe_rows[0]["target_log_value"], "nan")

    def test_dataset_preserves_quaternary_components(self):
        from salt_mix.dataset import load_encoded_split

        sample = {
            "record_id": "Q1",
            "system": "MgCl2-CaCl2-NaCl-KCl",
            "component_tokens": [["MgCl2"], ["CaCl2"], ["NaCl"], ["KCl"]],
            "mole_fractions": [0.0496, 0.0327, 0.5144, 0.4033],
            "temperature": 950.0,
            "target_value": 1.5,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            for split in ("train", "val", "test"):
                (artifact_dir / f"{split}_samples.json").write_text(json.dumps([sample]), encoding="utf-8")
            (artifact_dir / "vocab.json").write_text(
                json.dumps({"[PAD]": 0, "[UNK]": 1, "MgCl2": 2, "CaCl2": 3, "NaCl": 4, "KCl": 5}),
                encoding="utf-8",
            )

            encoded = load_encoded_split(artifact_dir, "test")[0]

        self.assertEqual(encoded["component_mask"], [1, 1, 1, 1])
        self.assertEqual(encoded["mole_fractions"], [0.0496, 0.0327, 0.5144, 0.4033])
        self.assertEqual(encoded["temperature"], 950.0)
        self.assertEqual(encoded["raw_temperature"], 950.0)
        self.assertNotIn("temperature_scaled", encoded)
        self.assertEqual(len(encoded["token_ids"]), 4)

    def test_dataset_standardizes_temperature_from_train_summary(self):
        from salt_mix.dataset import load_encoded_split

        train_sample = {
            "record_id": "D1",
            "system": "NaCl",
            "component_tokens": [["NaCl"]],
            "mole_fractions": [1.0],
            "temperature": 900.0,
            "target_value": 1.0,
        }
        test_sample = {
            "record_id": "D2",
            "system": "KCl",
            "component_tokens": [["KCl"]],
            "mole_fractions": [1.0],
            "temperature": 1100.0,
            "target_value": 2.0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "train_samples.json").write_text(json.dumps([train_sample]), encoding="utf-8")
            (artifact_dir / "val_samples.json").write_text(json.dumps([train_sample]), encoding="utf-8")
            (artifact_dir / "test_samples.json").write_text(json.dumps([test_sample]), encoding="utf-8")
            (artifact_dir / "vocab.json").write_text(
                json.dumps({"[PAD]": 0, "[UNK]": 1, "NaCl": 2, "KCl": 3}),
                encoding="utf-8",
            )
            (artifact_dir / "summary.json").write_text(
                json.dumps({"temperature_mean": 1000.0, "temperature_std": 100.0}),
                encoding="utf-8",
            )

            encoded = load_encoded_split(artifact_dir, "test")[0]

        self.assertEqual(encoded["raw_temperature"], 1100.0)
        self.assertAlmostEqual(encoded["temperature"], 1.0)


if __name__ == "__main__":
    unittest.main()
