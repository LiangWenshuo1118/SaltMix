from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, Tuple


ALLOWED_OXYANION_MARKERS: Tuple[str, ...] = (
    "NO2",
    "NO3",
    "CO3",
    "HCO3",
    "SO3",
    "SO4",
    "S2O7",
    "PO3",
    "PO4",
    "P2O7",
    "BO2",
    "B4O7",
    "MOO4",
    "WO4",
    "CRO4",
    "VO3",
    "VO4",
    "CLO3",
    "CLO4",
    "ALO2",
    "SIO3",
    "SIO4",
    "OH",
)

POLYATOMIC_ANIONS: Sequence[Tuple[str, int, str]] = (
    ("AlF6", -3, "AlF6^3-"),
    ("AlCl4", -1, "AlCl4^-"),
    ("AlBr4", -1, "AlBr4^-"),
    ("ZrF6", -2, "ZrF6^2-"),
    ("BF4", -1, "BF4^-"),
    ("PF6", -1, "PF6^-"),
    ("ClO4", -1, "ClO4^-"),
    ("ClO3", -1, "ClO3^-"),
    ("NO3", -1, "NO3^-"),
    ("NO2", -1, "NO2^-"),
    ("CO3", -2, "CO3^2-"),
    ("HCO3", -1, "HCO3^-"),
    ("SO4", -2, "SO4^2-"),
    ("SO3", -2, "SO3^2-"),
    ("S2O7", -2, "S2O7^2-"),
    ("PO4", -3, "PO4^3-"),
    ("PO3", -1, "PO3^-"),
    ("P2O7", -4, "P2O7^4-"),
    ("BO2", -1, "BO2^-"),
    ("B4O7", -2, "B4O7^2-"),
    ("MoO4", -2, "MoO4^2-"),
    ("WO4", -2, "WO4^2-"),
    ("CrO4", -2, "CrO4^2-"),
    ("VO3", -1, "VO3^-"),
    ("VO4", -3, "VO4^3-"),
    ("SiO3", -2, "SiO3^2-"),
    ("OH", -1, "OH^-"),
    ("SCN", -1, "SCN^-"),
    ("CN", -1, "CN^-"),
)

MONATOMIC_ANION_CHARGES: Dict[str, int] = {
    "F": -1,
    "Cl": -1,
    "Br": -1,
    "I": -1,
}

KNOWN_POLYCATIONS: Sequence[Tuple[str, int, str]] = (
    ("NH4", 1, "NH4^+"),
)


@dataclass
class ComponentRecord:
    formula: str
    tokens: List[str]
    parser_warning: bool
    metadata: Dict[str, str]


def normalize_formula(formula: str) -> str:
    return formula.replace("*", "-").strip()


def composition_dict(formula: str) -> Dict[str, int]:
    tokens = re.findall(r"[A-Z][a-z]?|\d+|\(|\)", formula)
    if "".join(tokens) != formula:
        raise ValueError(f"Unsupported formula syntax: {formula}")

    stack: List[Dict[str, int]] = [{}]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "(":
            stack.append({})
            index += 1
            continue
        if token == ")":
            if len(stack) == 1:
                raise ValueError(f"Unbalanced parentheses in {formula}")
            group = stack.pop()
            index += 1
            multiplier = 1
            if index < len(tokens) and tokens[index].isdigit():
                multiplier = int(tokens[index])
                index += 1
            for element, amount in group.items():
                stack[-1][element] = stack[-1].get(element, 0) + amount * multiplier
            continue
        if re.fullmatch(r"[A-Z][a-z]?", token):
            element = token
            index += 1
            count = 1
            if index < len(tokens) and tokens[index].isdigit():
                count = int(tokens[index])
                index += 1
            stack[-1][element] = stack[-1].get(element, 0) + count
            continue
        raise ValueError(f"Unexpected token {token} in {formula}")

    if len(stack) != 1:
        raise ValueError(f"Unbalanced parentheses in {formula}")
    return stack[0]


def subtract_counts(lhs: Dict[str, int], rhs: Dict[str, int], multiplier: int) -> Optional[Dict[str, int]]:
    result = dict(lhs)
    for element, amount in rhs.items():
        result[element] = result.get(element, 0) - amount * multiplier
        if result[element] < 0:
            return None
        if result[element] == 0:
            del result[element]
    return result


def contains_hydrocarbon(formula: str) -> bool:
    composition = composition_dict(formula)
    if "C" not in composition or "H" not in composition:
        return False
    upper = formula.upper()
    if any(marker in upper for marker in ("HCO3", "CO3", "CN", "SCN")):
        return False
    return True


def is_non_inorganic(formula: str) -> bool:
    upper = formula.upper()
    if contains_hydrocarbon(formula):
        return True
    if any(marker in upper for marker in ("C2H3O2", "CH3COO", "C6H5", "C3H7", "C4H9", "C5H11")):
        return True
    return False


def is_oxide_like(formula: str) -> bool:
    composition = composition_dict(formula)
    if "O" not in composition:
        return False
    upper = formula.upper()
    if any(marker in upper for marker in ALLOWED_OXYANION_MARKERS):
        return False
    return True


def is_sulfide_like(formula: str) -> bool:
    composition = composition_dict(formula)
    if "S" not in composition:
        return False
    upper = formula.upper()
    if any(marker in upper for marker in ("SO3", "SO4", "S2O7", "SCN")):
        return False
    return True


def classify_component(formula: str) -> Optional[str]:
    normalized = normalize_formula(formula)
    if is_non_inorganic(normalized):
        return "non_inorganic"
    if is_oxide_like(normalized):
        return "oxide_like"
    if is_sulfide_like(normalized):
        return "sulfide_like"
    return None


def infer_monoatomic_cation(remaining: Dict[str, int], total_positive_charge: int) -> Optional[Tuple[str, int, str, int]]:
    if len(remaining) != 1:
        return None
    element, count = next(iter(remaining.items()))
    if count <= 0 or total_positive_charge % count != 0:
        return None
    charge = total_positive_charge // count
    return element, count, f"{element}{charge}+", charge


def infer_known_polycation(remaining: Dict[str, int], total_positive_charge: int) -> Optional[Tuple[str, int, str, int]]:
    for formula, charge, token in KNOWN_POLYCATIONS:
        try:
            candidate = composition_dict(formula)
        except Exception:
            continue
        counts = []
        for element, amount in candidate.items():
            if element not in remaining or remaining[element] % amount != 0:
                counts = []
                break
            counts.append(remaining[element] // amount)
        if counts and len(set(counts)) == 1:
            stoich = counts[0]
            if stoich * charge == total_positive_charge:
                return formula, stoich, token, charge
    return None


def decompose_component(formula: str) -> ComponentRecord:
    normalized = normalize_formula(formula)
    base_tokens = ["[CLS]", "[COMP]", normalized]
    reason = classify_component(normalized)
    if reason is not None:
        return ComponentRecord(
            formula=normalized,
            tokens=base_tokens + ["[CAT]", "[UNK_CAT]", "[CAT_NUM]", "0", "[ANI]", "[UNK_ANI]", "[ANI_NUM]", "0", "[SEP]"],
            parser_warning=True,
            metadata={"reason": reason},
        )

    full_counts = composition_dict(normalized)
    candidates: List[Tuple[int, List[str], Dict[str, str]]] = []

    for anion_formula, anion_charge, anion_token in POLYATOMIC_ANIONS:
        try:
            anion_counts = composition_dict(anion_formula)
        except Exception:
            continue
        max_multiplier = min(full_counts.get(element, 0) // amount for element, amount in anion_counts.items())
        for multiplier in range(max_multiplier, 0, -1):
            remaining = subtract_counts(full_counts, anion_counts, multiplier)
            if not remaining:
                continue
            total_positive = -anion_charge * multiplier
            cation = infer_monoatomic_cation(remaining, total_positive) or infer_known_polycation(remaining, total_positive)
            if cation is None:
                continue
            cat_formula, cat_count, cat_token, _ = cation
            tokens = base_tokens + [
                "[CAT]",
                cat_token,
                "[CAT_NUM]",
                str(cat_count),
                "[ANI]",
                anion_token,
                "[ANI_NUM]",
                str(multiplier),
                "[SEP]",
            ]
            candidates.append((len(anion_formula), tokens, {"cation": cat_formula, "anion": anion_formula}))

    for symbol, charge in MONATOMIC_ANION_CHARGES.items():
        if symbol not in full_counts:
            continue
        multiplier = full_counts[symbol]
        remaining = subtract_counts(full_counts, {symbol: 1}, multiplier)
        if not remaining:
            continue
        total_positive = -charge * multiplier
        cation = infer_monoatomic_cation(remaining, total_positive) or infer_known_polycation(remaining, total_positive)
        if cation is None:
            continue
        cat_formula, cat_count, cat_token, _ = cation
        tokens = base_tokens + [
            "[CAT]",
            cat_token,
            "[CAT_NUM]",
            str(cat_count),
            "[ANI]",
            f"{symbol}-",
            "[ANI_NUM]",
            str(multiplier),
            "[SEP]",
        ]
        candidates.append((1, tokens, {"cation": cat_formula, "anion": symbol}))

    if candidates:
        _, tokens, metadata = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
        return ComponentRecord(formula=normalized, tokens=tokens, parser_warning=False, metadata=metadata)

    return ComponentRecord(
        formula=normalized,
        tokens=base_tokens + ["[CAT]", "[UNK_CAT]", "[CAT_NUM]", "0", "[ANI]", "[UNK_ANI]", "[ANI_NUM]", "0", "[SEP]"],
        parser_warning=True,
        metadata={"reason": "decomposition_failed"},
    )


def tokenize_components(components: Sequence[str]) -> List[ComponentRecord]:
    return [decompose_component(component) for component in components]
