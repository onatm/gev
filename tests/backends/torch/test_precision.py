import json
from dataclasses import replace
import pytest

from gev.configuration.config import load_config
from gev.backends.torch.masks import row_attention_mask
from gev.backends.torch.gemma3 import build_tiny_model
from gev.diagnostics.precision import check_precision
from gev.backends.torch.precision import compare_precision_profiles
import torch


def _encoded():
    return {"ids": [2, 3, 4, 5, 6, 7, 8], "seg": [0, 0, 1, 1, 1, 1, 1],
            "pos": [0, 1, 2, 3, 4, 5, 6], "opt": [-1, -1, -1, -1, 0, 1, -2],
            "decide_idx": [6], "opt_idx": [[4, 5]], "labels": [0], "state_length": 2}


def test_explicit_row_masks_have_causal_and_sliding_contract():
    positions = torch.tensor([[0, 1, 2, -1]])
    full = row_attention_mask(positions, "full_attention")
    sliding = row_attention_mask(positions, "sliding_attention", window=2)
    assert full.shape == (1, 1, 4, 4)
    assert bool(full[0, 0, 2, 0]) and not bool(full[0, 0, 0, 1])
    assert not bool(full[0, 0, 2, 3]) and not bool(sliding[0, 0, 2, 0])


def test_mps_fast_config_is_real_configuration():
    config = load_config("configs/gemma3-1b-mps-fast.toml")
    assert config.model.revision and config.training.logical_batch == 8


def test_precision_comparison_uses_actual_eager_and_sdpa_paths():
    result = compare_precision_profiles(build_tiny_model(), [_encoded()], ["tiny/1"],
                                        device="cpu", include_bf16=False)
    assert [(item["attention"], item["dtype"]) for item in result] == [("eager", "fp32"), ("sdpa", "fp32")]
    assert all(item["status"] == "passed" for item in result)


def test_precision_probabilities_are_raw_and_stable_id_keyed():
    result = compare_precision_profiles(build_tiny_model(), [_encoded()], ["stable-id"],
                                        device="cpu", include_bf16=False)
    assert set(result[0]["probabilities"]) == {"stable-id:0"}
    assert result[1]["compared_questions"] == 1


def test_precision_reports_measured_delta_kl_and_flips_not_placeholders():
    result = compare_precision_profiles(build_tiny_model(), [_encoded()], ["tiny"],
                                        device="cpu", include_bf16=False)[1]
    assert result["max_abs_probability"] is not None
    assert result["kl_divergence"] is not None
    assert result["argmax_flips"] == 0


def test_precision_checks_real_backward_gradients():
    result = compare_precision_profiles(build_tiny_model(), [_encoded()], ["tiny"],
                                        device="cpu", include_bf16=False)
    assert all(item["finite_gradient"] is True for item in result)


def test_runtime_precision_fields_are_explicit():
    config = load_config("configs/gemma3-1b-v7.toml")
    assert config.runtime.attn_implementation == "eager"
    assert config.runtime.gradient_checkpointing is False
    assert config.runtime.empty_cache is False


def test_precision_rejects_unresolved_backend_before_model_or_data_access(tmp_path):
    config = load_config("configs/smoke.toml")
    config = replace(config, backend=replace(config.backend, id="mlx"))
    result = check_precision(config, tmp_path / "precision.json", run="missing-run",
                             data_root=str(tmp_path / "no-data"))
    assert result["status"] == "failed"
    assert "unknown model backend" in result["error"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_precision_bf16_path_is_actual_forward_backward():
    result = compare_precision_profiles(build_tiny_model().to("mps"), [_encoded()], ["tiny"],
                                        device="mps", include_bf16=True)
    bf16 = next(item for item in result if item["dtype"] == "bf16")
    assert bf16["status"] == "passed"
    assert bf16["finite_gradient"] is True
