

import unittest

from deep_research.artifacts.metadata import (
    build_artifact_metadata,
)


class ArtifactMetadataTests(unittest.TestCase):
    def test_title_cannot_escape_final_directory(self):
        metadata = build_artifact_metadata(
            "../../.env",
            artifact_id="abcdef1234567890",
        )

        self.assertEqual(
            metadata["filename"],
            "env-abcdef123456.md",
        )

        self.assertEqual(
            metadata["workspace_path"],
            "/final/env-abcdef123456.md",
        )

        self.assertNotIn(
            "..",
            metadata["workspace_path"],
        )

    def test_two_artifacts_have_different_filenames(self):
        first = build_artifact_metadata(
            "研究报告",
            artifact_id="111111111111aaaa",
        )

        second = build_artifact_metadata(
            "研究报告",
            artifact_id="222222222222bbbb",
        )

        self.assertNotEqual(
            first["filename"],
            second["filename"],
        )

        self.assertTrue(
            first["filename"].endswith(".md")
        )

        self.assertTrue(
            second["filename"].endswith(".md")
        )

    def test_empty_title_is_rejected(self):
        with self.assertRaises(ValueError):
            build_artifact_metadata("   ")


if __name__ == "__main__":
    unittest.main()