import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RagProductionBoundaryTests(unittest.TestCase):
    def test_production_modules_do_not_import_evaluation(self):
        paths = [
            PROJECT_ROOT / "deep_research" / "rag",
            PROJECT_ROOT / "deep_research" / "tools",
            PROJECT_ROOT / "deep_research" / "handlers",
            PROJECT_ROOT / "deep_research" / "main.py",
            PROJECT_ROOT / "deep_research" / "app_lifecycle.py",
        ]
        for path in paths:
            candidates = [path] if path.is_file() else path.rglob("*.py")
            for candidate in candidates:
                source = candidate.read_text(encoding="utf-8")
                self.assertNotIn(
                    "deep_research.evaluation",
                    source,
                    msg=str(candidate),
                )
                self.assertNotIn(
                    "..evaluation",
                    source,
                    msg=str(candidate),
                )


if __name__ == "__main__":
    unittest.main()
