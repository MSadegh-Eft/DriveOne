import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class VerifyScriptTests(unittest.TestCase):
    def test_empty_data_root_is_not_ready_without_network_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/verify_refav.py"), "--data-root", directory],
                text=True,
                capture_output=True,
                check=True,
            )
        self.assertIn('"status": "not_ready"', result.stdout)
        self.assertIn("No local RefAV record file was found", result.stdout)


if __name__ == "__main__":
    unittest.main()
