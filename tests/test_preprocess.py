from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from salt_mix.chemistry import classify_component, decompose_component
from salt_mix.preprocess import preprocess_from_config
from salt_mix.inference import _build_subsystem_coverage_report, _build_subsystem_index, _prediction_rows
from salt_mix.properties import (
    canonical_property_unit,
    evaluate_property_targets,
    parse_arrhenius_equation,
    parse_composition,
    parse_composition_polynomial_equation,
    parse_constant_equation,
    parse_log10_arrhenius_equation,
    parse_linear_equation,
    parse_temperature_point,
    parse_temperature_range,
)


class PreprocessTests(unittest.TestCase):
    def test_parse_linear_equation_scientific(self):
        a, b = parse_linear_equation("d = 2.4968 - 3.9594 x 10^-4 T")
        self.assertAlmostEqual(a, 2.4968)
        self.assertAlmostEqual(b, 3.9594e-4)

    def test_parse_linear_equation_decimal(self):
        a, b = parse_linear_equation("d = 4.0791 - 0.0014529 T")
        self.assertAlmostEqual(a, 4.0791)
        self.assertAlmostEqual(b, 0.0014529)

    def test_parse_surface_tension_linear_equation(self):
        a, b = parse_linear_equation("g = 171.3 - 0.025 T")
        self.assertAlmostEqual(a, 171.3)
        self.assertAlmostEqual(b, 0.025)

    def test_parse_arrhenius_equation(self):
        a, b = parse_arrhenius_equation("v = 0.3806 exp (12920.40816/RT)")
        self.assertAlmostEqual(a, 0.3806)
        self.assertAlmostEqual(b, 12920.40816)

    def test_parse_electrical_conductivity_arrhenius_equation(self):
        a, b = parse_arrhenius_equation("k = 5.183 exp(-3476.96217/RT)")
        self.assertAlmostEqual(a, 5.183)
        self.assertAlmostEqual(b, -3476.96217)

    def test_electrical_conductivity_unit_is_reported_as_s_per_cm(self):
        self.assertEqual(canonical_property_unit("electrical_conductivity", "ohm^-1 cm^-1"), "S/cm")
        self.assertEqual(canonical_property_unit("electrical_conductivity", "S cm^-1"), "S/cm")

    def test_parse_log10_arrhenius_equation(self):
        a, b, c = parse_log10_arrhenius_equation("v = 10^(-0.545 + 597/T + 546000/T^2)")
        self.assertAlmostEqual(a, -0.545)
        self.assertAlmostEqual(b, 597.0)
        self.assertAlmostEqual(c, 546000.0)

    def test_evaluate_log10_arrhenius_viscosity(self):
        target_value, target_log_value = evaluate_property_targets(
            "viscosity",
            {"kind": "log10_arrhenius", "a": -0.545, "b": 597.0, "c": 546000.0},
            1000.0,
        )
        expected_value = 10 ** (-0.545 + 597.0 / 1000.0 + 546000.0 / 1000.0**2)
        self.assertAlmostEqual(target_value, expected_value)
        self.assertAlmostEqual(target_log_value, math.log(expected_value))

    def test_parse_constant_equation(self):
        value = parse_constant_equation("k = 2.83")
        self.assertAlmostEqual(value, 2.83)

    def test_parse_composition_polynomial_equation(self):
        coefficients = parse_composition_polynomial_equation("k = 5.852 x 10^-5 + 0.001428 C + 0.00158 C^2")
        self.assertEqual(coefficients[0], 5.852e-5)
        self.assertEqual(coefficients[1], 0.001428)
        self.assertEqual(coefficients[2], 0.00158)

    def test_parse_temperature_range(self):
        lo, hi = parse_temperature_range("720-940 K")
        self.assertEqual((lo, hi), (720.0, 940.0))

    def test_parse_temperature_point(self):
        value = parse_temperature_point("1000 K")
        self.assertEqual(value, 1000.0)

    def test_parse_composition(self):
        values = parse_composition("20-40-40 mol%", 3)
        self.assertEqual(values, [0.2, 0.4, 0.4])

    def test_prediction_rows_accept_empty_equation(self):
        rows = [
            {
                "index": "PRED_1",
                "data": {
                    "system": "MgCl2-CaCl2-NaCl-KCl",
                    "composition": "4.96-3.27-51.44-40.33 mol%",
                    "equation": "",
                    "temperature": "973 K",
                    "unit": "",
                },
            }
        ]

        samples = _prediction_rows(rows, "viscosity", temperatures_per_record=5)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["record_id"], "PRED_1")
        self.assertEqual(samples[0]["system"], "MgCl2-CaCl2-NaCl-KCl")
        self.assertEqual(samples[0]["temperature"], 973.0)
        self.assertEqual(samples[0]["mole_fractions"], [0.0496, 0.0327, 0.5144, 0.4033])
        self.assertIn("target_log_value", samples[0])

    def test_prediction_rows_expand_temperature_range_from_config_count(self):
        rows = [
            {
                "index": "PRED_RANGE",
                "data": {
                    "system": "NaCl-KCl",
                    "composition": "50-50 mol%",
                    "equation": "",
                    "temperature": "900-1000 K",
                    "unit": "",
                },
            }
        ]

        samples = _prediction_rows(rows, "density", temperatures_per_record=3)

        self.assertEqual([sample["temperature"] for sample in samples], [900.0, 950.0, 1000.0])
        self.assertTrue(all("target_log_value" not in sample for sample in samples))

    def test_subsystem_coverage_report_summarizes_training_data(self):
        coverage_samples = [
            {
                "record_id": "R_NACL",
                "system": "NaCl",
                "components": ["NaCl"],
                "mole_fractions": [1.0],
                "temperature": 1000.0,
            },
            {
                "record_id": "R_KCL_NACL",
                "system": "KCl-NaCl",
                "components": ["KCl", "NaCl"],
                "mole_fractions": [0.25, 0.75],
                "temperature": 950.0,
            },
            {
                "record_id": "R_KCL_NACL",
                "system": "KCl-NaCl",
                "components": ["KCl", "NaCl"],
                "mole_fractions": [0.25, 0.75],
                "temperature": 1000.0,
            },
            {
                "record_id": "R_TERNARY",
                "system": "KCl-MgCl2-NaCl",
                "components": ["KCl", "MgCl2", "NaCl"],
                "mole_fractions": [0.4, 0.1, 0.5],
                "temperature": 900.0,
            },
        ]
        prediction_sample = {
            "record_id": "PRED_1",
            "system": "MgCl2-CaCl2-NaCl-KCl",
            "components": ["MgCl2", "CaCl2", "NaCl", "KCl"],
            "mole_fractions": [0.0496, 0.0327, 0.5144, 0.4033],
        }

        report = _build_subsystem_coverage_report(
            prediction_sample,
            _build_subsystem_index(coverage_samples),
        )

        self.assertEqual(report["subsystem_count"], 14)
        self.assertEqual(report["covered_subsystem_count"], 3)
        self.assertIn("MgCl2-CaCl2", report["missing_subsystems"])
        self.assertIn("3/14 lower-order subsystems covered", report["summary"])

        nacl_kcl = next(item for item in report["subsystems"] if item["subsystem"] == "NaCl-KCl")
        self.assertTrue(nacl_kcl["covered"])
        self.assertEqual(nacl_kcl["systems"], ["KCl-NaCl"])
        self.assertEqual(nacl_kcl["temperature_range_K"], {"min": 950.0, "max": 1000.0})
        self.assertEqual(
            nacl_kcl["composition_ranges_mol_pct"],
            {"NaCl": {"min_mol_pct": 75.0, "max_mol_pct": 75.0}, "KCl": {"min_mol_pct": 25.0, "max_mol_pct": 25.0}},
        )

    def test_subsystem_coverage_report_aligns_composition_to_requested_order(self):
        coverage_samples = [
            {
                "record_id": "R_TERNARY",
                "system": "KCl-MgCl2-NaCl",
                "components": ["KCl", "MgCl2", "NaCl"],
                "mole_fractions": [0.4, 0.1, 0.5],
                "temperature": 900.0,
            },
        ]
        prediction_sample = {
            "record_id": "PRED_1",
            "system": "MgCl2-CaCl2-NaCl-KCl",
            "components": ["MgCl2", "CaCl2", "NaCl", "KCl"],
            "mole_fractions": [0.0496, 0.0327, 0.5144, 0.4033],
        }

        report = _build_subsystem_coverage_report(
            prediction_sample,
            _build_subsystem_index(coverage_samples),
        )

        ternary = next(item for item in report["subsystems"] if item["subsystem"] == "MgCl2-NaCl-KCl")
        self.assertEqual(ternary["systems"], ["KCl-MgCl2-NaCl"])
        self.assertAlmostEqual(
            ternary["query_subsystem_composition_mol_pct"]["MgCl2"],
            100.0 * 0.0496 / (0.0496 + 0.5144 + 0.4033),
            places=3,
        )
        self.assertEqual(
            ternary["composition_points_mol_pct"]["values"],
            [{"MgCl2": 10.0, "NaCl": 50.0, "KCl": 40.0}],
        )

    def test_filtering(self):
        self.assertEqual(classify_component("N(C3H7)4B(C6H5)4"), "non_inorganic")
        self.assertEqual(classify_component("N(C5H11)4SCN"), "non_inorganic")
        self.assertEqual(classify_component("Al2O3"), "oxide_like")
        self.assertEqual(classify_component("Ag2S"), "sulfide_like")
        self.assertIsNone(classify_component("CsCl"))
        self.assertIsNone(classify_component("NH4Cl"))
        self.assertIsNone(classify_component("NaHCO3"))
        self.assertIsNone(classify_component("NaNO3"))
        self.assertIsNone(classify_component("NaOH"))

    def test_decompose_component(self):
        item = decompose_component("Mg(NO3)2")
        self.assertFalse(item.parser_warning)
        self.assertIn("Mg2+", item.tokens)
        self.assertIn("NO3^-", item.tokens)

    def test_decompose_complex_fluoride(self):
        item = decompose_component("Na3AlF6")
        self.assertFalse(item.parser_warning)
        self.assertIn("Na1+", item.tokens)
        self.assertIn("AlF6^3-", item.tokens)

    def test_decompose_metasilicate(self):
        item = decompose_component("CaSiO3")
        self.assertFalse(item.parser_warning)
        self.assertIn("Ca2+", item.tokens)
        self.assertIn("SiO3^2-", item.tokens)
        self.assertNotIn("[UNK_CAT]", item.tokens)
        self.assertNotIn("[UNK_ANI]", item.tokens)

    def test_decompose_tetrahaloaluminate_salts(self):
        cases = [
            ("KAlCl4", "K1+", "AlCl4^-"),
            ("LiAlBr4", "Li1+", "AlBr4^-"),
            ("NaAlCl4", "Na1+", "AlCl4^-"),
        ]
        for formula, cation, anion in cases:
            with self.subTest(formula=formula):
                item = decompose_component(formula)
                self.assertFalse(item.parser_warning)
                self.assertIn(cation, item.tokens)
                self.assertIn(anion, item.tokens)
                self.assertNotIn("[UNK_CAT]", item.tokens)
                self.assertNotIn("[UNK_ANI]", item.tokens)

    def test_formula_parser_with_parentheses(self):
        item = decompose_component("Ba(NO3)2")
        self.assertFalse(item.parser_warning)
        self.assertIn("Ba2+", item.tokens)
        self.assertIn("NO3^-", item.tokens)

    def test_viscosity_preprocess_uses_page_glob_with_dedup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "viscosity" / "raw"
            raw_dir.mkdir(parents=True)
            page_rows_a = [
                {
                    "index": "V1",
                    "source": {"doi": ""},
                    "data": {
                        "system": "AgBr",
                        "composition": "100 mol%",
                        "equation": "v = 0.3806 exp (12920.40816/RT)",
                        "temperature": "713-873 K",
                        "unit": "mN s m^-2",
                    },
                },
                {
                    "index": "V2",
                    "source": {"doi": ""},
                    "data": {
                        "system": "AgBr-KBr",
                        "composition": "50-50 mol%",
                        "equation": "v = 0.3000 exp (10000/RT)",
                        "temperature": "700-900 K",
                        "unit": "mN s m^-2",
                    },
                },
            ]
            page_rows_b = [
                {
                    "index": "V2",
                    "source": {"doi": ""},
                    "data": {
                        "system": "AgBr-KBr",
                        "composition": "50-50 mol%",
                        "equation": "v = 0.3000 exp (10000/RT)",
                        "temperature": "700-900 K",
                        "unit": "mN s m^-2",
                    },
                },
                {
                    "index": "V3",
                    "source": {"doi": ""},
                    "data": {
                        "system": "AgBr-KBr-NaBr",
                        "composition": "40-30-30 mol%",
                        "equation": "v = 0.2500 exp (9000/RT)",
                        "temperature": "700-900 K",
                        "unit": "mN s m^-2",
                    },
                },
                        {
                            "index": "BAD_NON_ARRHENIUS",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgCl",
                                "composition": "100 mol%",
                                "equation": "v = 4.56",
                                "temperature": "700 K",
                                "unit": "mN s m^-2",
                            },
                        }
            ]
            (raw_dir / "viscosity_raw.page272.json").write_text(json.dumps(page_rows_a), encoding="utf-8")
            (raw_dir / "viscosity_raw.page273.json").write_text(json.dumps(page_rows_b), encoding="utf-8")
            artifact_dir = root / "outputs" / "viscosity"
            config = root / "viscosity.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_viscosity_dir: {raw_dir}",
                                                "  validation_ratio: 0.5",
                        "  temperatures_per_record: 2",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "viscosity")

            train_samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8"))
            val_samples = json.loads((artifact_dir / "val_samples.json").read_text(encoding="utf-8"))
            test_samples = json.loads((artifact_dir / "test_samples.json").read_text(encoding="utf-8"))
            all_samples = train_samples + val_samples + test_samples

            self.assertEqual(summary["normalized_records"], 4)
            self.assertTrue(any(sample["record_id"] == "BAD_NON_ARRHENIUS" for sample in all_samples))
            self.assertTrue(all("target_log_value" in sample for sample in all_samples))
            self.assertEqual(len(test_samples), 2)

    def test_viscosity_preprocess_filters_unsupported_temperature_formula(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_file = root / "viscosity_raw.page272.json"
            raw_file.write_text(
                json.dumps(
                    [
                        {
                            "index": "GOOD",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr",
                                "composition": "100 mol%",
                                "equation": "v = 0.3806 exp (12920.40816/RT)",
                                "temperature": "713-873 K",
                                "unit": "mN s m^-2",
                            },
                        },
                        {
                            "index": "BAD",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr",
                                "composition": "100 mol%",
                                "equation": "v = 1.0 + 0.01 T",
                                "temperature": "713-873 K",
                                "unit": "mN s m^-2",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifacts"
            config = root / "viscosity.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_viscosity_dir: {raw_file.parent}",
                                                "  validation_ratio: 0.5",
                        "  temperatures_per_record: 1",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "viscosity")
            filter_report = json.loads((artifact_dir / "filter_report.json").read_text(encoding="utf-8"))
            samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8")) + json.loads(
                (artifact_dir / "val_samples.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary["normalized_records"], 1)
            self.assertEqual(filter_report[0]["record_id"], "BAD")
            self.assertEqual(len(samples), 1)

    def test_viscosity_preprocess_excludes_configured_systems(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_file = root / "viscosity_raw.page272.json"
            raw_file.write_text(
                json.dumps(
                    [
                        {
                            "index": "KEEP",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr",
                                "composition": "100 mol%",
                                "equation": "v = 0.3806 exp (12920.40816/RT)",
                                "temperature": "713-873 K",
                                "unit": "mN s m^-2",
                            },
                        },
                        {
                            "index": "DROP",
                            "source": {"doi": ""},
                            "data": {
                                "system": "BeF2-LiF",
                                "composition": "50-50 mol%",
                                "equation": "v = 0.00845 exp (58702.5021/RT)",
                                "temperature": "660-840 K",
                                "unit": "mN s m^-2",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifacts"
            config = root / "viscosity.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_viscosity_dir: {raw_file.parent}",
                                                "  validation_ratio: 0.5",
                        "  temperatures_per_record: 1",
                        "  excluded_systems:",
                        "    - BeF2-LiF",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "viscosity")
            filter_report = json.loads((artifact_dir / "filter_report.json").read_text(encoding="utf-8"))
            samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8")) + json.loads(
                (artifact_dir / "val_samples.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary["normalized_records"], 1)
            self.assertEqual(summary["excluded_systems"], ["BeF2-LiF"])
            self.assertEqual(filter_report[0]["record_id"], "DROP")
            self.assertEqual(filter_report[0]["reason"], "excluded_system")
            self.assertEqual(len(samples), 1)

    def test_random_point_split_allows_system_overlap_between_train_and_val(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_file = root / "viscosity_raw.page272.json"
            raw_file.write_text(
                json.dumps(
                    [
                        {
                            "index": "R1",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr-KBr",
                                "composition": "50-50 mol%",
                                "equation": "v = 0.3000 exp (10000/RT)",
                                "temperature": "700-900 K",
                                "unit": "mN s m^-2",
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifacts"
            config = root / "viscosity.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_viscosity_dir: {raw_file.parent}",
                                                "  validation_ratio: 0.5",
                        "  temperatures_per_record: 5",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "viscosity")
            train_samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8"))
            val_samples = json.loads((artifact_dir / "val_samples.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["property"], "viscosity")

    def test_record_level_split_keeps_expanded_points_together(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_file = root / "viscosity_raw.page272.json"
            raw_file.write_text(
                json.dumps(
                    [
                        {
                            "index": "R1",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr-KBr",
                                "composition": "50-50 mol%",
                                "equation": "v = 0.3000 exp (10000/RT)",
                                "temperature": "700-900 K",
                                "unit": "mN s m^-2",
                            },
                        },
                        {
                            "index": "R2",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgCl-KCl",
                                "composition": "50-50 mol%",
                                "equation": "v = 0.2800 exp (9500/RT)",
                                "temperature": "720-920 K",
                                "unit": "mN s m^-2",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifacts"
            config = root / "viscosity.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_viscosity_dir: {raw_file.parent}",
                        "  validation_ratio: 0.5",
                        "  temperatures_per_record: 3",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "viscosity")
            train_samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8"))
            val_samples = json.loads((artifact_dir / "val_samples.json").read_text(encoding="utf-8"))
            train_ids = {sample["record_id"] for sample in train_samples}
            val_ids = {sample["record_id"] for sample in val_samples}

            self.assertEqual(summary["train_records"], 1)
            self.assertEqual(summary["val_records"], 1)
            self.assertEqual(len(train_samples), 3)
            self.assertEqual(len(val_samples), 3)
            self.assertTrue(train_ids.isdisjoint(val_ids))

    def test_interval_expansion_uses_temperature_and_composition_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "density"
            raw_dir.mkdir(parents=True)
            (raw_dir / "density_raw.page001.json").write_text(
                json.dumps(
                    [
                        {
                            "index": "POLY_RANGE",
                            "source": {"doi": ""},
                            "data": {
                                "system": "NaCl-KCl",
                                "composition": "0-12 NaCl",
                                "equation": "d = 1.0 + 0.01 C",
                                "temperature": "700-820 K",
                                "unit": "g cm^-3",
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifacts"
            config = root / "density.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_density_dir: {raw_dir}",
                        "  validation_ratio: 0.5",
                        "  temperatures_per_record: 2",
                        "  temperature_interval: 50",
                        "  composition_interval: 5",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "density")
            samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8"))
            temperatures = sorted({sample["temperature"] for sample in samples})
            composition_values = sorted({round(sample["mole_fractions"][0] * 100.0, 8) for sample in samples})

            self.assertEqual(summary["temperature_interval"], 50.0)
            self.assertEqual(summary["composition_interval"], 5.0)
            self.assertEqual(temperatures, [700.0, 750.0, 800.0, 820.0])
            self.assertEqual(composition_values, [0.0, 5.0, 10.0, 12.0])

    def test_low_data_learning_uses_prefix_of_seeded_train_split_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_file = root / "viscosity_raw.page272.json"
            raw_file.write_text(
                json.dumps(
                    [
                        {
                            "index": "R1",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr-KBr",
                                "composition": "50-50 mol%",
                                "equation": "v = 0.3000 exp (10000/RT)",
                                "temperature": "700-900 K",
                                "unit": "mN s m^-2",
                            },
                        },
                        {
                            "index": "R2",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgCl-KCl",
                                "composition": "50-50 mol%",
                                "equation": "v = 0.2800 exp (9500/RT)",
                                "temperature": "720-920 K",
                                "unit": "mN s m^-2",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            full_artifact_dir = root / "artifacts_full"
            full_config = root / "viscosity_full.yaml"
            full_config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_viscosity_dir: {raw_file.parent}",
                        "  validation_ratio: 0.2",
                        "  temperatures_per_record: 5",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  low_data_learning: false",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {full_artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            low_artifact_dir = root / "artifacts_low"
            low_config = root / "viscosity_low.yaml"
            low_config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_viscosity_dir: {raw_file.parent}",
                        "  validation_ratio: 0.2",
                        "  temperatures_per_record: 5",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  low_data_learning: true,0.5",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {low_artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            preprocess_from_config(full_config, "viscosity")
            low_summary = preprocess_from_config(low_config, "viscosity")

            full_train_samples = json.loads((full_artifact_dir / "train_samples.json").read_text(encoding="utf-8"))
            full_val_samples = json.loads((full_artifact_dir / "val_samples.json").read_text(encoding="utf-8"))
            low_train_samples = json.loads((low_artifact_dir / "train_samples.json").read_text(encoding="utf-8"))
            low_val_samples = json.loads((low_artifact_dir / "val_samples.json").read_text(encoding="utf-8"))

            expected_train_count = int(round(len(full_train_samples) * 0.5))
            self.assertEqual(low_summary["train_samples_before_low_data"], len(full_train_samples))
            self.assertTrue(low_summary["low_data_learning"])
            self.assertEqual(low_summary["low_data_learning_ratio"], 0.5)
            self.assertEqual(low_summary["train_samples"], expected_train_count)
            self.assertEqual(low_train_samples, full_train_samples[:expected_train_count])
            self.assertEqual(low_val_samples, full_val_samples)

    def test_quaternary_records_are_included_in_test_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_file = root / "viscosity_raw.page272.json"
            raw_file.write_text(
                json.dumps(
                    [
                        {
                            "index": "Q4",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr-KBr-NaBr-LiBr",
                                "composition": "25-25-25-25 mol%",
                                "equation": "v = 0.3000 exp (10000/RT)",
                                "temperature": "700-900 K",
                                "unit": "mN s m^-2",
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifacts"
            config = root / "viscosity.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_viscosity_dir: {raw_file.parent}",
                        "  validation_ratio: 0.5",
                        "  temperatures_per_record: 2",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "viscosity")
            train_samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8"))
            val_samples = json.loads((artifact_dir / "val_samples.json").read_text(encoding="utf-8"))
            test_samples = json.loads((artifact_dir / "test_samples.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["normalized_records"], 1)
            self.assertEqual(len(train_samples), 0)
            self.assertEqual(len(val_samples), 0)
            self.assertEqual(len(test_samples), 2)

    def test_surface_tension_preprocess_accepts_linear_constant_and_fixed_temp_polynomial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "surface_tension"
            raw_dir.mkdir(parents=True)
            (raw_dir / "surface_tension_raw.page114.json").write_text(
                json.dumps(
                    [
                        {
                            "index": "GOOD_ST",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr",
                                "composition": "100 mol%",
                                "equation": "g = 171.3 - 0.025 T",
                                "temperature": "733-893 K",
                                "unit": "mN m^-1",
                            },
                        },
                        {
                            "index": "CONST_ST",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr",
                                "composition": "100 mol%",
                                "equation": "g = 690",
                                "temperature": "733 K",
                                "unit": "mN m^-1",
                            },
                        },
                        {
                            "index": "POLY_ST",
                            "source": {"doi": ""},
                            "data": {
                                "system": "KF-Na2B4O7",
                                "composition": "100-40 Na2B4O7 mol%",
                                "equation": "g = 146.2 - 1.193 C + 0.02033 C^2",
                                "temperature": "1223 K",
                                "unit": "mN m^-1",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifacts"
            config = root / "surface_tension.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_surface_tension_dir: {raw_dir}",
                        "  validation_ratio: 0.5",
                        "  temperatures_per_record: 1",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "surface_tension")
            filter_report = json.loads((artifact_dir / "filter_report.json").read_text(encoding="utf-8"))
            samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8")) + json.loads(
                (artifact_dir / "val_samples.json").read_text(encoding="utf-8")
            )
            samples_by_id = {sample["record_id"]: sample for sample in samples}

            self.assertEqual(summary["normalized_records"], 3)
            self.assertEqual(filter_report, [])
            self.assertEqual(len(samples), 3)
            self.assertNotIn("target_log_value", samples_by_id["GOOD_ST"])
            self.assertNotIn("target_log_value", samples_by_id["CONST_ST"])
            self.assertAlmostEqual(samples_by_id["CONST_ST"]["target_value"], 690.0)
            self.assertAlmostEqual(samples_by_id["POLY_ST"]["mole_fractions"][0], 0.3)
            self.assertAlmostEqual(samples_by_id["POLY_ST"]["mole_fractions"][1], 0.7)

    def test_density_preprocess_accepts_constant_and_fixed_temp_polynomial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "density"
            raw_dir.mkdir(parents=True)
            (raw_dir / "density_raw.page047.json").write_text(
                json.dumps(
                    [
                        {
                            "index": "CONST_DEN",
                            "source": {"doi": ""},
                            "data": {
                                "system": "HgBr2-HgI2",
                                "composition": "0-100 mol%",
                                "equation": "d = 5.34",
                                "temperature": "531.2 K",
                                "unit": "g cm^-3",
                            },
                        },
                        {
                            "index": "POLY_DEN",
                            "source": {"doi": ""},
                            "data": {
                                "system": "BaF2-KCl",
                                "composition": "0-100 BaF2 mol%",
                                "equation": "d = 1.199 + 0.03076 C - 6.245 x 10^-5 C^2 + 3.029 x 10^-7 C^3",
                                "temperature": "1273 K",
                                "unit": "g cm^-3",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifacts"
            config = root / "density.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_density_dir: {raw_dir}",
                        "  validation_ratio: 0.5",
                        "  temperatures_per_record: 1",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "density")
            filter_report = json.loads((artifact_dir / "filter_report.json").read_text(encoding="utf-8"))
            samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8")) + json.loads(
                (artifact_dir / "val_samples.json").read_text(encoding="utf-8")
            )
            samples_by_id = {sample["record_id"]: sample for sample in samples}

            self.assertEqual(summary["normalized_records"], 2)
            self.assertEqual(filter_report, [])
            self.assertAlmostEqual(samples_by_id["CONST_DEN"]["target_value"], 5.34)
            self.assertAlmostEqual(samples_by_id["POLY_DEN"]["mole_fractions"][0], 0.5)
            self.assertAlmostEqual(samples_by_id["POLY_DEN"]["mole_fractions"][1], 0.5)

    def test_viscosity_preprocess_accepts_constant_and_fixed_temp_polynomial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "viscosity"
            raw_dir.mkdir(parents=True)
            (raw_dir / "viscosity_raw.page300.json").write_text(
                json.dumps(
                    [
                        {
                            "index": "CONST_VIS",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgI-AgNO3",
                                "composition": "0-100 mol%",
                                "equation": "v = 4.56",
                                "temperature": "498 K",
                                "unit": "mN s m^-2",
                            },
                        },
                        {
                            "index": "POLY_VIS",
                            "source": {"doi": ""},
                            "data": {
                                "system": "Na2SO4-Na4P2O7",
                                "composition": "30-100 Na2SO4",
                                "equation": "v = 22.987 - 0.2732 C + 9.3 x 10^-4 C^2",
                                "temperature": "1273 K",
                                "unit": "mN s m^-2",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifacts"
            config = root / "viscosity.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_viscosity_dir: {raw_dir}",
                        "  validation_ratio: 0.5",
                        "  temperatures_per_record: 1",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "viscosity")
            filter_report = json.loads((artifact_dir / "filter_report.json").read_text(encoding="utf-8"))
            samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8")) + json.loads(
                (artifact_dir / "val_samples.json").read_text(encoding="utf-8")
            )
            samples_by_id = {sample["record_id"]: sample for sample in samples}

            self.assertEqual(summary["normalized_records"], 2)
            self.assertEqual(filter_report, [])
            self.assertIn("target_log_value", samples_by_id["CONST_VIS"])
            self.assertIn("target_log_value", samples_by_id["POLY_VIS"])
            self.assertAlmostEqual(samples_by_id["POLY_VIS"]["mole_fractions"][0], 0.65)
            self.assertAlmostEqual(samples_by_id["POLY_VIS"]["mole_fractions"][1], 0.35)

    def test_electrical_conductivity_preprocess_accepts_arrhenius_constant_and_fixed_temp_polynomial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "electrical_conductivity"
            raw_dir.mkdir(parents=True)
            (raw_dir / "electrical_conductivity_raw.page164.json").write_text(
                json.dumps(
                    [
                        {
                            "index": "GOOD_EC",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr",
                                "composition": "100 mol%",
                                "equation": "k = 5.183 exp(-3476.96217/RT)",
                                "temperature": "723-1073 K",
                                "unit": "ohm^-1 cm^-1",
                            },
                        },
                        {
                            "index": "CONST_EC_POINT",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr-NaBr",
                                "composition": "50-50 mol%",
                                "equation": "k = 2.83",
                                "temperature": "1000 K",
                                "unit": "ohm^-1 cm^-1",
                            },
                        },
                        {
                            "index": "POLY_EC_FIXED_TEMP",
                            "source": {"doi": ""},
                            "data": {
                                "system": "BeF2-Na3AlF6",
                                "composition": "100-55 mol% Na3AlF6",
                                "equation": "k = 5.852 x 10^-5 + 0.001428 C + 0.00158 C^2",
                                "temperature": "1273 K",
                                "unit": "ohm^-1 cm^-1",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifacts"
            config = root / "electrical_conductivity.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_electrical_conductivity_dir: {raw_dir}",
                        "  validation_ratio: 0.5",
                        "  temperatures_per_record: 1",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "electrical_conductivity")
            filter_report = json.loads((artifact_dir / "filter_report.json").read_text(encoding="utf-8"))
            samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8")) + json.loads(
                (artifact_dir / "val_samples.json").read_text(encoding="utf-8")
            )
            samples_by_id = {sample["record_id"]: sample for sample in samples}

            self.assertEqual(summary["normalized_records"], 3)
            self.assertEqual(filter_report, [])
            self.assertEqual(len(samples), 3)
            self.assertIn("target_log_value", samples_by_id["GOOD_EC"])
            self.assertTrue(all(sample["unit"] == "S/cm" for sample in samples))

            const_sample = samples_by_id["CONST_EC_POINT"]
            self.assertEqual(const_sample["temperature"], 1000.0)
            self.assertEqual(const_sample["mole_fractions"], [0.5, 0.5])
            self.assertAlmostEqual(const_sample["target_value"], 2.83)
            self.assertAlmostEqual(const_sample["target_log_value"], 1.040276711655146)

            poly_sample = samples_by_id["POLY_EC_FIXED_TEMP"]
            self.assertEqual(poly_sample["temperature"], 1273.0)
            self.assertAlmostEqual(poly_sample["mole_fractions"][0], 0.225)
            self.assertAlmostEqual(poly_sample["mole_fractions"][1], 0.775)
            expected_value, expected_log = evaluate_property_targets(
                "electrical_conductivity",
                {"kind": "composition_poly", "coefficients": {0: 5.852e-5, 1: 0.001428, 2: 0.00158}},
                1273.0,
                composition_value=77.5,
            )
            self.assertAlmostEqual(poly_sample["target_value"], expected_value)
            self.assertAlmostEqual(poly_sample["target_log_value"], expected_log)

    def test_electrical_conductivity_preprocess_filters_non_positive_fixed_temp_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "electrical_conductivity"
            raw_dir.mkdir(parents=True)
            (raw_dir / "electrical_conductivity_raw.page170.json").write_text(
                json.dumps(
                    [
                        {
                            "index": "ZERO_CONST",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AlBr3-SbBr3",
                                "composition": "50-50 mol%",
                                "equation": "k = 0.0",
                                "temperature": "373 K",
                                "unit": "ohm^-1 cm^-1",
                            },
                        },
                        {
                            "index": "GOOD_CONST",
                            "source": {"doi": ""},
                            "data": {
                                "system": "AgBr-NaBr",
                                "composition": "50-50 mol%",
                                "equation": "k = 2.83",
                                "temperature": "1000 K",
                                "unit": "ohm^-1 cm^-1",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifacts"
            config = root / "electrical_conductivity.yaml"
            config.write_text(
                "\n".join(
                    [
                        "data:",
                        f"  raw_electrical_conductivity_dir: {raw_dir}",
                        "  validation_ratio: 0.5",
                        "  temperatures_per_record: 1",
                        "model:",
                        "  embed_dim: 16",
                        "  encoder_dim: 16",
                        "  num_heads: 4",
                        "  num_layers: 1",
                        "  ff_dim: 32",
                        "  subnet_dim: 16",
                        "  rk_order: 1",
                        "  dropout: 0.1",
                        "training:",
                        "  seed: 7",
                        "  batch_size: 2",
                        "  learning_rate: 0.001",
                        "  epochs: 1",
                        f"  output_dir: {artifact_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            summary = preprocess_from_config(config, "electrical_conductivity")
            filter_report = json.loads((artifact_dir / "filter_report.json").read_text(encoding="utf-8"))
            samples = json.loads((artifact_dir / "train_samples.json").read_text(encoding="utf-8")) + json.loads(
                (artifact_dir / "val_samples.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary["normalized_records"], 2)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["record_id"], "GOOD_CONST")
            self.assertEqual(filter_report[0]["record_id"], "ZERO_CONST")
            self.assertEqual(filter_report[0]["reason"], "no_valid_samples")


if __name__ == "__main__":
    unittest.main()
