from pathlib import Path

import pytest

from drenavision.core import config


@pytest.mark.parametrize("absolute", [False, True])
def test_legacy_workspace_paths_follow_the_folder_migration(
    tmp_path, monkeypatch, absolute
):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    legacy = Path("artifacts/models/best.pt")
    settings = config.Settings(
        _env_file=None,
        model_path=tmp_path / legacy if absolute else legacy,
        database_path=Path("artifacts/events/drenavision.db"),
        evidence_dir=Path("artifacts/events/evidence"),
    )
    assert settings.model_path == tmp_path / "resultados/local/models/best.pt"
    assert settings.database_path == tmp_path / "resultados/local/events/drenavision.db"
    assert settings.evidence_dir == tmp_path / "resultados/local/events/evidence"


def test_legacy_migration_preserves_existing_and_external_paths(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(config, "PROJECT_ROOT", project)
    existing = project / "artifacts/models/best.pt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing-custom-checkpoint")
    external = tmp_path / "another_project/artifacts/events/drenavision.db"
    settings = config.Settings(
        _env_file=None, model_path=existing, database_path=external
    )
    assert settings.model_path == existing
    assert settings.database_path == external
