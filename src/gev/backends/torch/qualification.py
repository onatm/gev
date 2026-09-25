"""Small production checks specific to the text-only Torch Gemma 4 path."""

from __future__ import annotations

import torch


def collect_torch_training_checks(model, training_metrics: dict) -> dict[str, bool]:
    """Collect strict source, precision, optimizer, and update evidence."""
    source = getattr(model, "source_provenance", None) or {}
    source_exact = (
        source.get("model") == "google/gemma-4-E2B"
        and source.get("revision") == "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
        and source.get("source_weights_dtype") == "bf16"
        and source.get("source_text_tensor_count") == 600
        and source.get("source_text_parameters") == 4_647_449_891
        and source.get("effective_text_parameters") == 4_628_569_344
        and isinstance(source.get("source_name_shape_sha256"), str)
        and len(source["source_name_shape_sha256"]) == 64)
    compute_dtype = getattr(model, "compute_dtype", None)
    expected_dtype = torch.bfloat16 if compute_dtype == "bf16" else torch.float32
    frozen = [parameter for parameter in model.backbone.parameters()
              if not parameter.requires_grad]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    engineering = training_metrics.get("qualification_checks", {})
    return {
        "source_inventory_exact": source_exact,
        "decoder_compute_dtype": bool(frozen) and all(
            parameter.dtype == expected_dtype for parameter in frozen),
        "fp32_lora_and_pointer": bool(trainable) and all(
            parameter.dtype == torch.float32 for parameter in trainable),
        "fp32_optimizer_state": engineering.get("fp32_optimizer_state") is True,
        "finite_loss_gradients": engineering.get("finite_loss_gradients") is True,
        "nonzero_trainable_update": engineering.get("nonzero_trainable_update") is True,
    }


def causal_attention_masks_valid(model, encodings: list[dict], *, compute_dtype: str) -> bool:
    """Confirm future-token mutations cannot affect earlier decoder states."""
    config = getattr(model.decoder, "config", None)
    layer_types = set(getattr(config, "layer_types", ()))
    if not {"full_attention", "sliding_attention"} <= layer_types or not encodings:
        return False
    was_training = model.training
    model.eval()
    compute_type = torch.bfloat16 if compute_dtype == "bf16" else torch.float32
    tolerance = (4 if compute_dtype == "bf16" else 32) * torch.finfo(compute_type).eps
    try:
        with torch.inference_mode():
            for encoded in encodings[:4]:
                ids = torch.as_tensor(encoded["ids"], dtype=torch.long,
                                      device=next(model.parameters()).device).unsqueeze(0)
                positions = torch.as_tensor(encoded["pos"], dtype=torch.long,
                                             device=ids.device).unsqueeze(0)
                if ids.shape[1] < 2:
                    return False
                changed = ids.clone()
                vocab_size = int(config.vocab_size)
                changed[0, -1] = (changed[0, -1] + 1) % vocab_size
                mask = torch.ones_like(ids)
                original = model.decoder(
                    input_ids=ids, attention_mask=mask, position_ids=positions,
                    use_cache=False).last_hidden_state
                mutated = model.decoder(
                    input_ids=changed, attention_mask=mask, position_ids=positions,
                    use_cache=False).last_hidden_state
                if not torch.isfinite(original).all() or not torch.isfinite(mutated).all():
                    return False
                if not torch.allclose(original[:, :-1].float(), mutated[:, :-1].float(),
                                      atol=tolerance, rtol=tolerance):
                    return False
        return True
    finally:
        model.train(was_training)
