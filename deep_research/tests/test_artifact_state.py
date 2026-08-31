import unittest

from deep_research.state.research import (
    merge_artifacts,
)


def artifact(
    artifact_id: str,
    filename: str,
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "filename": filename,
        "workspace_path": f"/final/{filename}",
        "created_at": "2026-08-20T12:00:00+00:00",
        "size_bytes": 12,
        "sha256": "a" * 64,
    }


class ArtifactStateTests(unittest.TestCase):
    def test_reducer_merges_artifacts_without_mutating_inputs(self):
        current = {
            "a" * 32: artifact(
                "a" * 32,
                "first.md",
            )
        }

        update = {
            "b" * 32: artifact(
                "b" * 32,
                "second.md",
            )
        }

        merged = merge_artifacts(
            current,
            update,
        )

        self.assertEqual(
            set(merged),
            {"a" * 32, "b" * 32},
        )
        self.assertEqual(
            merged["b" * 32]["filename"],
            "second.md",
        )
        self.assertEqual(
            set(current),
            {"a" * 32},
        )

    def test_reducer_updates_same_artifact_id(self):
        artifact_id = "a" * 32

        current = {
            artifact_id: artifact(
                artifact_id,
                "old.md",
            )
        }

        update = {
            artifact_id: artifact(
                artifact_id,
                "new.md",
            )
        }

        merged = merge_artifacts(
            current,
            update,
        )

        self.assertEqual(
            merged[artifact_id]["filename"],
            "new.md",
        )


if __name__ == "__main__":
    unittest.main()