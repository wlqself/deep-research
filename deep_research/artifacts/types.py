from typing_extensions import TypedDict


class ArtifactMetadata(TypedDict):
    artifact_id: str
    filename: str
    workspace_path: str
    created_at: str
    size_bytes: int
    sha256: str