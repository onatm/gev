import os
import hashlib
import json

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import pytest

pytest.importorskip("mlx.core", reason="MLX qualification tests require the optional MLX extra on Apple Silicon")
pytest.importorskip("mlx_lm", reason="MLX qualification tests require the optional mlx-lm package")

import numpy as np

from gev.backends.mlx.qualification import (
    BOUNDARY_IDS,
    POLICY,
    POLICY_SHA256,
    SELECTED_LAYER_INDICES,
    SELECTED_IDS_SHA256,
    SELECTION_ALGORITHM,
    _all_boundary_tables_exact,
    _attention_masks_exact,
    _compare_selected_layers,
    _exact_boundary_match,
    _source_inventory_matches,
    compare_outputs,
    qualification_code_sha256,
    selected_ids_sha256,
    select_development_rows,
    require_tf32_disabled_before_mlx_import,
    seal_receipt,
)


def _row(identifier, source, body, questions=1):
    return {"_meta": {"id": identifier, "source": source},
            "state": body,
            "questions": {f"q{index}": {"instructions": body}
                          for index in range(questions)}}


def test_development_selection_is_source_balanced_deterministic_and_length_stratified():
    rows = [
        _row("a-short", "a", "x"), _row("a-long", "a", "x" * 50),
        _row("a-middle", "a", "x" * 20, questions=2),
        _row("b-short", "b", "y"), _row("b-long", "b", "y" * 40),
        _row("c-short", "c", "z"), _row("c-long", "c", "z" * 30),
    ]
    selected = select_development_rows(rows)
    again = select_development_rows(list(reversed(rows)))
    selected_ids = [row["_meta"]["id"] for row in selected]

    assert selected_ids == [row["_meta"]["id"] for row in again]
    assert set(selected_ids) == {"a-short", "a-long", "a-middle",
                                 "b-short", "b-long", "c-short", "c-long"}
    assert len([row for row in selected if row["_meta"]["source"] == "a"]) == 3
    assert any(len(row["questions"]) > 1 for row in selected)
    assert selected_ids_sha256(selected) == selected_ids_sha256(again)
    assert SELECTION_ALGORITHM == "source-balanced-shortest-longest-canonical-json-v1"


def test_fp32_cross_backend_diagnostic_is_separate_from_trained_precision_policy():
    from gev.models.policy import GEMMA4_POLICY

    assert POLICY.source_weights_dtype == "bf16"
    assert POLICY.gate_for("fp32").max_abs_probability is None
    assert POLICY.gate_for("fp32").require_argmax_agreement is False
    assert POLICY.gate_for("fp32").gate_kind == "independent_implementation_diagnostic"
    assert "fp32_lora_and_pointer" in GEMMA4_POLICY.required_qualification_checks(
        "mlx", "fp32")
    assert "development_coverage" in GEMMA4_POLICY.required_qualification_checks(
        "torch", "bf16")
    assert POLICY.mlx_tf32_disabled is True
    assert POLICY.selected_layer_indices == SELECTED_LAYER_INDICES == (0, 14, 34)
    assert BOUNDARY_IDS == (6, 7, 8, 9, 10, 239673, 239674, 239675, 239676, 262143)
    assert POLICY.selected_ids_sha256 == SELECTED_IDS_SHA256
    assert POLICY.sha256() == POLICY_SHA256


def test_tf32_guard_requires_environment_before_an_already_imported_mlx(monkeypatch):
    import sys
    import pytest

    monkeypatch.setitem(sys.modules, "mlx.core", object())
    monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    with pytest.raises(RuntimeError, match="before importing mlx.core"):
        require_tf32_disabled_before_mlx_import()
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    require_tf32_disabled_before_mlx_import()


def test_receipt_code_identity_includes_backend_neutral_training_batching(tmp_path):
    sources = (
        "src/gev/backends/mlx/gemma4.py", "src/gev/backends/mlx/__init__.py",
        "src/gev/backends/mlx/pointer.py",
        "src/gev/backends/mlx/qualification.py", "src/gev/backends/mlx/training.py",
        "src/gev/backends/mlx/checkpoint.py", "src/gev/domain/tokenization.py",
        "src/gev/training/batching.py", "src/gev/training/schedule.py",
        "src/gev/training/policy.py", "src/gev/models/specs.py",
        "src/gev/models/policy.py",
        "src/gev/models/registry.py", "src/gev/models/families.py", "models.lock.json",
        "src/gev/configuration/config.py", "src/gev/configuration/resolved.py",
        "src/gev/application/training.py", "src/gev/application/evaluation.py",
        "src/gev/application/prediction.py", "src/gev/commands/diagnose.py",
        "src/gev/evaluation/development.py",
    )
    for name in sources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    before = qualification_code_sha256(tmp_path)
    batching = tmp_path / "src/gev/training/batching.py"
    batching.write_text("changed logical-to-physical token accounting", encoding="utf-8")
    assert qualification_code_sha256(tmp_path) != before


def test_fp32_diagnostic_receipt_is_content_addressed_and_fails_closed_on_staleness():
    base = seal_receipt({
        "protocol": POLICY.protocol,
        "policy_sha256": POLICY_SHA256,
        "status": "base_structure_passed",
        "inference_qualified": False,
        "base": {"name": POLICY.base_model, "revision": POLICY.base_revision,
                 "family": POLICY.family, "backend": POLICY.backend,
                 "source_weights_dtype": "bf16", "compute_dtype": "fp32"},
        "code_sha256": "a" * 64,
        "hardware": {"machine": "arm64"},
        "stages": {"fp32_structure": {
            "status": "passed", "source_inventory_exact": True,
            "mlx_source_inventory": {"text_tensor_count": 600},
            "decoder_contract_exact": True,
            "boundary_gathers_exact": {"main": True, "ple": True},
            "attention_masks_exact": True,
            "selected_layer_outputs": {"match": True},
            "probability_outputs": {"finite": True},
        }},
    })
    payload = {key: value for key, value in base.items() if key != "receipt_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    assert base["receipt_sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    tampered = {key: value for key, value in base.items() if key != "receipt_sha256"}
    tampered["status"] = "failed"
    assert seal_receipt(tampered)["receipt_sha256"] != base["receipt_sha256"]


def test_output_comparison_rejects_nonfinite_and_shape_mismatches():
    reference = {
        "hidden": {"row::q0": {"decision": np.array([1.0]),
                               "options": np.array([[2.0], [3.0]])}},
        "heads": {"0": {"row::q0": {"logits": np.array([1.0, 2.0]),
                                      "probabilities": np.array([0.3, 0.7])}}},
    }
    candidate = {
        "hidden": {"row::q0": {"decision": np.array([1.0]),
                               "options": np.array([[2.0], [3.0]])}},
        "heads": {"0": {"row::q0": {"logits": np.array([1.0, 2.0]),
                                      "probabilities": np.array([0.3, 0.7])}}},
    }
    assert compare_outputs(reference, candidate)["finite"] is True
    trained_reference = {**reference, "heads": {"trained": reference["heads"]["0"]}}
    trained_candidate = {**candidate, "heads": {"trained": candidate["heads"]["0"]}}
    trained_metrics = compare_outputs(trained_reference, trained_candidate)
    assert trained_metrics["trained_head"] is True
    assert trained_metrics["pointer_head_seeds"] == []

    candidate["heads"]["0"]["row::q0"]["logits"] = np.array([np.nan, 2.0])
    assert compare_outputs(reference, candidate)["finite"] is False

    candidate["heads"]["0"]["row::q0"]["logits"] = np.array([1.0])
    assert compare_outputs(reference, candidate)["status"] == "failed"


def test_fp32_implementation_diagnostic_reports_weight_inventory_and_output_differences():
    torch_inventory = {
        "source_tensor_count": 2011, "source_text_tensor_count": 600,
        "source_text_parameters": 4_647_449_891,
        "source_name_shape_sha256": "a" * 64,
        "effective_text_parameters": 4_628_569_344,
        "source_revision": POLICY.base_revision,
    }
    mlx_inventory = {
        "tensor_count": 2011, "text_tensor_count": 600,
        "text_serialized_parameters": 4_647_449_891,
        "name_shape_sha256": "a" * 64,
        "effective_text_parameters": 4_628_569_344,
    }
    assert _source_inventory_matches(torch_inventory, mlx_inventory)
    mlx_inventory["name_shape_sha256"] = "b" * 64
    assert not _source_inventory_matches(torch_inventory, mlx_inventory)

    reference = {
        "hidden": {"r::q0": {"decision": np.ones((2,)), "options": np.ones((2, 2))}},
        "heads": {"0": {"r::q0": {"logits": np.array([1.0, 0.0]),
                                    "probabilities": np.array([0.73, 0.27])}}},
    }
    wrong_weights = {
        "hidden": {"r::q0": {"decision": np.ones((2,)), "options": np.ones((2, 2))}},
        "heads": {"0": {"r::q0": {"logits": np.array([0.0, 1.0]),
                                    "probabilities": np.array([0.27, 0.73])}}},
    }
    metrics = compare_outputs(reference, wrong_weights)
    assert metrics["argmax_flips"] == 1
    assert metrics["probabilities_max_abs"] > 0
    assert metrics["finite"] is True


def test_engineering_gate_rejects_bad_masks_and_corrupt_high_boundary_row():
    masks = {length: {kind: {"match": True} for kind in
                      ("full_attention", "sliding_attention")}
             for length in ("short", "long")}
    assert _attention_masks_exact(masks)
    masks["long"]["sliding_attention"]["match"] = False
    assert not _attention_masks_exact(masks)

    reference = {"main": np.zeros((len(BOUNDARY_IDS), 2), dtype=np.float32),
                 "ple": np.zeros((len(BOUNDARY_IDS), 3), dtype=np.float32)}
    corrupted = {name: values.copy() for name, values in reference.items()}
    corrupted["ple"][-1, 0] = 1.0
    assert BOUNDARY_IDS[-1] == 262143
    matches = _exact_boundary_match(reference, corrupted)
    assert matches == {"main": True, "ple": False}
    assert not _all_boundary_tables_exact(matches)


@pytest.mark.parametrize("table", ("main", "ple"))
def test_boundary_mismatch_blocks_structural_and_precision_stage(table):
    reference = {"main": np.array([[1.0, 2.0]]), "ple": np.array([[3.0, 4.0]])}
    candidate = {name: values.copy() for name, values in reference.items()}
    candidate[table][0, 0] += 1.0
    matches = _exact_boundary_match(reference, candidate)
    assert matches == {"main": table != "main", "ple": table != "ple"}

    assert not _all_boundary_tables_exact(matches)
