from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, Optional, Tuple

R_GAS_CONSTANT = 8.31446261815324

LINEAR_EQUATION_RE = re.compile(
    r"""^\s*[a-zA-Z]\s*=\s*
    (?P<a>[+-]?(?:\d+(?:\.\d*)?|\.\d+))
    \s*(?P<sign>[+-])\s*
    (?P<b>(?:\d+(?:\.\d*)?|\.\d+)(?:\s*x\s*10\^(?:[+-]?\d+))?)
    \s*T\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
ARRHENIUS_EQUATION_RE = re.compile(
    r"""^\s*[a-zA-Z]\s*=\s*
    (?P<a>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*x\s*10\^(?:[+-]?\d+))?)
    \s*exp\s*\(\s*
    (?P<b>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*x\s*10\^(?:[+-]?\d+))?)
    \s*/\s*R\s*T\s*\)\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
LOG10_ARRHENIUS_EQUATION_RE = re.compile(
    r"""^\s*[a-zA-Z]\s*=\s*10\s*\^\s*\(\s*
    (?P<a>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*x\s*10\^(?:[+-]?\d+))?)
    \s*(?P<b_sign>[+-])\s*
    (?P<b>(?:\d+(?:\.\d*)?|\.\d+)(?:\s*x\s*10\^(?:[+-]?\d+))?)
    \s*/\s*T
    \s*(?P<c_sign>[+-])\s*
    (?P<c>(?:\d+(?:\.\d*)?|\.\d+)(?:\s*x\s*10\^(?:[+-]?\d+))?)
    \s*/\s*T\s*\^\s*2\s*
    \)\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
TEMP_RANGE_RE = re.compile(r"^\s*(?P<lo>\d+(?:\.\d+)?)\s*-\s*(?P<hi>\d+(?:\.\d+)?)\s*K\s*$", re.IGNORECASE)
TEMP_POINT_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*K\s*$", re.IGNORECASE)
CONSTANT_EQUATION_RE = re.compile(
    r"""^\s*[a-zA-Z]\s*=\s*
    (?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*x\s*10\^(?:[+-]?\d+))?)
    \s*$""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class PropertySpec:
    name: str
    raw_data_key: str
    training_target_key: str
    supports_log_metrics: bool
    equation_family: str
    display_unit: str
    unit_aliases: Tuple[str, ...] = ()


PROPERTY_REGISTRY: Dict[str, PropertySpec] = {
    "density": PropertySpec(
        name="density",
        raw_data_key="raw_density_dir",
        training_target_key="target_value",
        supports_log_metrics=False,
        equation_family="linear",
        display_unit="g/cm^3",
        unit_aliases=("g cm^-3",),
    ),
    "viscosity": PropertySpec(
        name="viscosity",
        raw_data_key="raw_viscosity_dir",
        training_target_key="target_log_value",
        supports_log_metrics=True,
        equation_family="arrhenius",
        display_unit="mPa s",
        unit_aliases=("mN s m^-2",),
    ),
    "surface_tension": PropertySpec(
        name="surface_tension",
        raw_data_key="raw_surface_tension_dir",
        training_target_key="target_value",
        supports_log_metrics=False,
        equation_family="linear",
        display_unit="mN/m",
        unit_aliases=("mN m^-1",),
    ),
    "electrical_conductivity": PropertySpec(
        name="electrical_conductivity",
        raw_data_key="raw_electrical_conductivity_dir",
        training_target_key="target_log_value",
        supports_log_metrics=True,
        equation_family="arrhenius",
        display_unit="S/cm",
        unit_aliases=("ohm^-1 cm^-1", "S cm^-1"),
    ),
}


def get_property_spec(property_name: str) -> PropertySpec:
    try:
        return PROPERTY_REGISTRY[property_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported property: {property_name}") from exc


def _normalize_unit_text(unit: str) -> str:
    return re.sub(r"\s+", " ", unit.strip()).lower()


def canonical_property_unit(property_name: str, unit: str = "") -> str:
    spec = get_property_spec(property_name)
    if not unit.strip():
        return spec.display_unit
    normalized_unit = _normalize_unit_text(unit)
    known_units = (spec.display_unit,) + spec.unit_aliases
    for known_unit in known_units:
        if normalized_unit == _normalize_unit_text(known_unit):
            return spec.display_unit
    return unit.strip()


def parse_scientific_literal(token: str) -> float:
    cleaned = token.strip().replace(" ", "")
    cleaned = cleaned.replace("x10^", "e")
    cleaned = cleaned.replace("X10^", "e")
    return float(cleaned)


def parse_linear_equation(equation: str) -> Tuple[float, float]:
    match = LINEAR_EQUATION_RE.match(equation)
    if not match:
        raise ValueError(f"Unsupported equation format: {equation}")
    a = float(match.group("a"))
    b = parse_scientific_literal(match.group("b"))
    if match.group("sign") == "+":
        b = -b
    return a, b


def parse_arrhenius_equation(equation: str) -> Tuple[float, float]:
    match = ARRHENIUS_EQUATION_RE.match(equation)
    if not match:
        raise ValueError(f"Unsupported equation format: {equation}")
    a = parse_scientific_literal(match.group("a"))
    b = parse_scientific_literal(match.group("b"))
    if a <= 0.0:
        raise ValueError(f"Arrhenius prefactor must be positive: {equation}")
    return a, b


def parse_log10_arrhenius_equation(equation: str) -> Tuple[float, float, float]:
    match = LOG10_ARRHENIUS_EQUATION_RE.match(equation)
    if not match:
        raise ValueError(f"Unsupported equation format: {equation}")
    a = parse_scientific_literal(match.group("a"))
    b = parse_scientific_literal(match.group("b"))
    c = parse_scientific_literal(match.group("c"))
    if match.group("b_sign") == "-":
        b = -b
    if match.group("c_sign") == "-":
        c = -c
    return a, b, c


def parse_temperature_range(raw: str) -> Tuple[float, float]:
    match = TEMP_RANGE_RE.match(raw)
    if not match:
        raise ValueError(f"Unsupported temperature range: {raw}")
    lo = float(match.group("lo"))
    hi = float(match.group("hi"))
    return lo, hi


def parse_temperature_point(raw: str) -> float:
    match = TEMP_POINT_RE.match(raw)
    if not match:
        raise ValueError(f"Unsupported temperature point: {raw}")
    return float(match.group("value"))


def parse_composition(raw: str, component_count: int) -> list[float]:
    text = raw.strip()
    if text.endswith("mol%"):
        text = text[:-4].strip()
    parts = [part.strip() for part in text.split("-")]
    if len(parts) != component_count:
        raise ValueError(f"Composition/component mismatch: {raw} vs {component_count}")
    values = [float(part) / 100.0 for part in parts]
    if abs(sum(values) - 1.0) > 1e-3:
        raise ValueError(f"Composition does not sum to 1: {raw}")
    return values


def parse_constant_equation(equation: str) -> float:
    match = CONSTANT_EQUATION_RE.match(equation)
    if not match:
        raise ValueError(f"Unsupported equation format: {equation}")
    return parse_scientific_literal(match.group("value"))


def parse_composition_polynomial_equation(equation: str) -> Dict[int, float]:
    if "=" not in equation:
        raise ValueError(f"Unsupported equation format: {equation}")
    _, rhs = equation.split("=", 1)
    normalized = rhs.replace(" ", "")
    normalized = normalized.replace("x10^", "e").replace("X10^", "e")
    term_pattern = re.compile(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?(?:\*?C(?:\^\d+)?)?",
        re.IGNORECASE,
    )
    terms = term_pattern.findall(normalized)
    if not terms or any("T" in term.upper() for term in terms):
        raise ValueError(f"Unsupported equation format: {equation}")
    if "".join(terms) != normalized:
        raise ValueError(f"Unsupported equation format: {equation}")

    coefficients: Dict[int, float] = {}
    for term in terms:
        if "C" not in term.upper():
            degree = 0
            coefficient = float(term)
        else:
            match = re.fullmatch(
                r"(?P<coef>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\*?C(?:\^(?P<degree>\d+))?",
                term,
                re.IGNORECASE,
            )
            if not match:
                raise ValueError(f"Unsupported equation format: {equation}")
            degree = int(match.group("degree") or "1")
            coefficient = float(match.group("coef"))
        coefficients[degree] = coefficients.get(degree, 0.0) + coefficient
    return coefficients


def evaluate_property_targets(
    property_name: str,
    equation_params: Dict[str, object],
    temperature: float,
    composition_value: Optional[float] = None,
) -> Tuple[float, Optional[float]]:
    spec = get_property_spec(property_name)
    if spec.equation_family == "linear":
        kind = str(equation_params.get("kind", "linear"))
        if kind == "linear":
            target_value = float(equation_params["a"]) - float(equation_params["b"]) * temperature
            return target_value, None
        if kind == "constant":
            return float(equation_params["value"]), None
        if kind == "composition_poly":
            if composition_value is None:
                raise ValueError("composition_value is required for composition polynomial equations")
            coefficients = equation_params["coefficients"]
            assert isinstance(coefficients, dict)
            target_value = sum(float(coefficient) * (composition_value ** int(degree)) for degree, coefficient in coefficients.items())
            return target_value, None
        raise ValueError(f"Unsupported equation kind: {kind}")
    if spec.equation_family == "arrhenius":
        kind = str(equation_params.get("kind", "arrhenius"))
        if kind == "arrhenius":
            log_value = math.log(float(equation_params["a"])) + float(equation_params["b"]) / (R_GAS_CONSTANT * temperature)
            return math.exp(log_value), log_value
        if kind == "log10_arrhenius":
            log10_value = (
                float(equation_params["a"])
                + float(equation_params["b"]) / temperature
                + float(equation_params["c"]) / (temperature ** 2)
            )
            log_value = math.log(10.0) * log10_value
            return math.exp(log_value), log_value
        if kind == "constant":
            target_value = float(equation_params["value"])
            if target_value <= 0.0:
                raise ValueError(f"Property value must be positive for log metrics: {target_value}")
            return target_value, math.log(target_value)
        if kind == "composition_poly":
            if composition_value is None:
                raise ValueError("composition_value is required for composition polynomial equations")
            coefficients = equation_params["coefficients"]
            assert isinstance(coefficients, dict)
            target_value = sum(float(coefficient) * (composition_value ** int(degree)) for degree, coefficient in coefficients.items())
            if target_value <= 0.0:
                raise ValueError(f"Property value must be positive for log metrics: {target_value}")
            return target_value, math.log(target_value)
        raise ValueError(f"Unsupported equation kind: {kind}")
    raise ValueError(f"Unsupported property: {property_name}")


def parse_property_equation(property_name: str, equation: str) -> Dict[str, object]:
    spec = get_property_spec(property_name)
    if spec.equation_family == "linear":
        try:
            a, b = parse_linear_equation(equation)
            return {"kind": "linear", "a": a, "b": b}
        except ValueError:
            pass
        try:
            value = parse_constant_equation(equation)
            return {"kind": "constant", "value": value}
        except ValueError:
            pass
        coefficients = parse_composition_polynomial_equation(equation)
        return {"kind": "composition_poly", "coefficients": coefficients}
    if spec.equation_family == "arrhenius":
        try:
            a, b = parse_arrhenius_equation(equation)
            return {"kind": "arrhenius", "a": a, "b": b}
        except ValueError:
            pass
        try:
            a, b, c = parse_log10_arrhenius_equation(equation)
            return {"kind": "log10_arrhenius", "a": a, "b": b, "c": c}
        except ValueError:
            pass
        try:
            value = parse_constant_equation(equation)
            return {"kind": "constant", "value": value}
        except ValueError:
            pass
        coefficients = parse_composition_polynomial_equation(equation)
        return {"kind": "composition_poly", "coefficients": coefficients}
    raise ValueError(f"Unsupported property: {property_name}")
