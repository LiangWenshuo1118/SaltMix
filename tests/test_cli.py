from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


class CliDispatchTests(unittest.TestCase):
    def test_salt_mix_dispatches_property(self):
        from salt_mix.__main__ import main

        with patch("salt_mix.preprocess.preprocess_from_config", return_value={"ok": True}) as mock_preprocess:
            with patch("builtins.print"):
                with patch.object(
                    sys,
                    "argv",
                    ["salt_mix", "preprocess", "--config", "demo.yaml", "--property", "viscosity"],
                ):
                    main()
        mock_preprocess.assert_called_once_with(Path("demo.yaml"), "viscosity")

    def test_salt_mix_dispatches_density(self):
        from salt_mix.__main__ import main

        with patch("salt_mix.preprocess.preprocess_from_config", return_value={"ok": True}) as mock_preprocess:
            with patch("builtins.print"):
                with patch.object(
                    sys,
                    "argv",
                    ["salt_mix", "preprocess", "--config", "demo.yaml", "--property", "density"],
                ):
                    main()
        mock_preprocess.assert_called_once_with(Path("demo.yaml"), "density")

    def test_salt_mix_dispatches_surface_tension(self):
        from salt_mix.__main__ import main

        with patch("salt_mix.preprocess.preprocess_from_config", return_value={"ok": True}) as mock_preprocess:
            with patch("builtins.print"):
                with patch.object(
                    sys,
                    "argv",
                    ["salt_mix", "preprocess", "--config", "demo.yaml", "--property", "surface_tension"],
                ):
                    main()
        mock_preprocess.assert_called_once_with(Path("demo.yaml"), "surface_tension")

    def test_salt_mix_dispatches_electrical_conductivity(self):
        from salt_mix.__main__ import main

        with patch("salt_mix.preprocess.preprocess_from_config", return_value={"ok": True}) as mock_preprocess:
            with patch("builtins.print"):
                with patch.object(
                    sys,
                    "argv",
                    ["salt_mix", "preprocess", "--config", "demo.yaml", "--property", "electrical_conductivity"],
                ):
                    main()
        mock_preprocess.assert_called_once_with(Path("demo.yaml"), "electrical_conductivity")


if __name__ == "__main__":
    unittest.main()
