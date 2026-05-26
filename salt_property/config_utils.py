from __future__ import annotations

TEMPERATURE_LAWS = {"none", "arrhenius", "linear"}


def normalize_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be true or false")


def normalize_ratio(value: object, *, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} must be a float between 0 and 1")
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a float between 0 and 1")
    ratio = float(value)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"{field_name} must be in the interval (0, 1]")
    return ratio


def resolve_low_data_learning(value: object) -> tuple[bool, float]:
    field_name = "low_data_learning"
    if value is None:
        return False, 1.0
    if isinstance(value, bool):
        return value, 1.0
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must be false or formatted as true,<ratio>")
        if "," not in normalized:
            enabled = normalize_bool(normalized, field_name=field_name)
            return enabled, 1.0
        raw_enabled, raw_ratio = [part.strip() for part in normalized.split(",", 1)]
        enabled = normalize_bool(raw_enabled, field_name=field_name)
        ratio = normalize_ratio(raw_ratio, field_name=field_name)
        return enabled, ratio
    raise ValueError(f"{field_name} must be false or formatted as true,<ratio>")


def normalize_rk_order(value: object) -> int:
    if value is None:
        raise ValueError("rk_order must be a non-negative integer")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("rk_order must be a non-negative integer")
        value = normalized
    if isinstance(value, bool):
        raise ValueError("rk_order must be a non-negative integer")
    rk_order = int(value)
    if rk_order < 0:
        raise ValueError("rk_order must be a non-negative integer")
    return rk_order


def resolve_model_rk_order(model_config: dict[str, object]) -> int | None:
    if not resolve_model_use_phy(model_config):
        return None
    if resolve_model_temperature_law(model_config) != "none":
        use_rk_value = model_config.get("use_rk", True)
        if normalize_bool(use_rk_value, field_name="use_rk"):
            raise ValueError("use_arrhenius/use_linear requires use_rk: false")
        return None
    use_rk_value = model_config.get("use_rk", True)
    if not normalize_bool(use_rk_value, field_name="use_rk"):
        return None
    return normalize_rk_order(model_config.get("rk_order", 2))


def resolve_model_use_phy(model_config: dict[str, object]) -> bool:
    return normalize_bool(model_config.get("use_phy", True), field_name="use_phy")


def normalize_temperature_law(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"", "false", "off", "no"}:
            return "none"
        if normalized in {"true"}:
            raise ValueError("temperature_law must be one of: none, arrhenius, linear")
        if normalized in {"no_temperature", "no_temperature_law"}:
            return "none"
        if normalized in TEMPERATURE_LAWS:
            return normalized
    raise ValueError("temperature_law must be one of: none, arrhenius, linear")


def resolve_model_temperature_law(model_config: dict[str, object]) -> str:
    explicit_law = "temperature_law" in model_config
    temperature_law = normalize_temperature_law(model_config.get("temperature_law"))

    use_arrhenius = normalize_bool(model_config.get("use_arrhenius", False), field_name="use_arrhenius")
    if "use_linear" in model_config and "use_linear_temperature" in model_config:
        use_linear = normalize_bool(model_config.get("use_linear"), field_name="use_linear")
        legacy_use_linear = normalize_bool(
            model_config.get("use_linear_temperature"),
            field_name="use_linear_temperature",
        )
        if use_linear != legacy_use_linear:
            raise ValueError("use_linear and use_linear_temperature must agree when both are set")
    else:
        use_linear = normalize_bool(
            model_config.get("use_linear", model_config.get("use_linear_temperature", False)),
            field_name="use_linear",
        )
    if use_arrhenius and use_linear:
        raise ValueError("use_arrhenius and use_linear cannot both be true")
    if explicit_law:
        if use_arrhenius and temperature_law != "arrhenius":
            raise ValueError("use_arrhenius: true requires temperature_law: arrhenius")
        if use_linear and temperature_law != "linear":
            raise ValueError("use_linear: true requires temperature_law: linear")
    elif use_arrhenius:
        temperature_law = "arrhenius"
    elif use_linear:
        temperature_law = "linear"

    if temperature_law != "none" and not resolve_model_use_phy(model_config):
        raise ValueError("use_arrhenius/use_linear requires use_phy: true")
    return temperature_law


def resolve_model_use_arrhenius(model_config: dict[str, object]) -> bool:
    return resolve_model_temperature_law(model_config) == "arrhenius"
