import inspect
import sys
from pathlib import Path

import yaml

from drenavision.data.converter import convert_dataset
from drenavision.inference.config import EvaluateConfig, TrainConfig
from drenavision.inference.evaluation import run_evaluation
from drenavision.inference.training import run_training
from ferramentas.scripts import (
    build_demo_video,
    evaluate,
    train,
    validate_dataset,
    visualize_annotations,
)


def test_current_entry_points_share_v2_ground_truth_and_safe_checkpoint_targets():
    train_args = train.build_parser().parse_args([])
    eval_args = evaluate.build_parser().parse_args([])
    train_values = yaml.safe_load(train_args.config.read_text(encoding="utf-8"))
    eval_values = yaml.safe_load(eval_args.config.read_text(encoding="utf-8"))
    assert (
        TrainConfig().data
        == EvaluateConfig().data
        == Path(train_values["data"])
        == Path(eval_values["data"])
    )
    assert inspect.signature(run_training).parameters["config_path"].default == str(
        train_args.config
    ).replace("\\", "/")
    assert inspect.signature(run_evaluation).parameters["config_path"].default == str(
        eval_args.config
    ).replace("\\", "/")
    assert (
        TrainConfig().published_checkpoint
        == Path(train_values["published_checkpoint"])
        == Path("resultados/local/models/candidate_v2.pt")
    )
    assert (
        EvaluateConfig().checkpoint
        == Path(eval_values["checkpoint"])
        == Path("resultados/local/models/best.pt")
    )
    valid_args = validate_dataset.build_parser().parse_args([])
    demo_args = build_demo_video.build_parser().parse_args([])
    assert (
        valid_args.dataset_root == demo_args.dataset_root == TrainConfig().data.parent
    )
    assert valid_args.yaml == TrainConfig().data
    assert (
        valid_args.manifest
        == demo_args.manifest
        == valid_args.dataset_root / "data_manifest.csv"
    )


def test_historical_configs_and_corrected_conversion_do_not_replace_active_model():
    historical_train = yaml.safe_load(
        Path("ferramentas/configs/train.yaml").read_text(encoding="utf-8")
    )
    historical_eval = yaml.safe_load(
        Path("ferramentas/configs/evaluate.yaml").read_text(encoding="utf-8")
    )
    assert (
        historical_train["published_checkpoint"]
        == "resultados/local/models/candidate_v1_reproduction.pt"
    )
    assert historical_eval["checkpoint"] == "resultados/local/models/best_v1.pt"
    signature = inspect.signature(convert_dataset)
    assert signature.parameters["output_root"].default == Path(
        "dados/processed/drenavision_corrected"
    )
    assert signature.parameters["manifest_path"].default == Path(
        "dados/manifests/dataset_manifest_corrected.csv"
    )


def test_annotation_preview_defaults_use_current_labels_and_local_outputs(monkeypatch):
    captured = {}
    monkeypatch.setattr(sys, "argv", ["visualize_annotations.py"])

    def fake_visualize(dataset_root, output_root, **_kwargs):
        captured.update(dataset_root=dataset_root, output_root=output_root)
        return [output_root / "preview.jpg"]

    monkeypatch.setattr(visualize_annotations, "visualize_dataset", fake_visualize)
    monkeypatch.setattr(
        visualize_annotations, "create_contact_sheet", lambda _paths, output: output
    )
    assert visualize_annotations.main() == 0
    assert captured["dataset_root"] == TrainConfig().data.parent
    assert captured["output_root"] == Path("resultados/local/dataset_preview_v2")
