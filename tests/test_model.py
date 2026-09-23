import torch
import pytest

from gev.models.gemma import build_tiny_model


def encoded(state=(2, 3), branches=((4, 5), (6,))):
    ids = list(state)
    pos = list(range(len(ids)))
    seg = [0] * len(ids)
    opt = [-1] * len(ids)
    decide, ends = [], []
    for question, options in enumerate(branches, 1):
        start = len(ids)
        branch_position = len(state)
        for option in options:
            ids.extend((10, option))
            pos.extend((branch_position, branch_position + 1))
            branch_position += 2
            seg.extend((question, question))
            opt.extend((len(ends), len(ends)))
        ids.append(20)
        pos.append(branch_position)
        seg.append(question)
        opt.append(-2)
        ends.append([start + 1, start + 3][:len(options)])
        decide.append(len(ids) - 1)
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt,
            "decide_idx": decide, "opt_idx": ends, "labels": [0] * len(branches),
            "state_length": len(state)}


def test_actual_tiny_gemma_rows_are_grouped_and_isolated():
    model = build_tiny_model()
    model.eval()
    first = encoded()
    alone = model.forward_one(first)
    together = model.forward_batch([first, encoded(state=(7, 8, 9), branches=((4,),))])
    assert len(alone) == 2 and [value.numel() for value in alone] == [2, 1]
    for expected, actual in zip(alone, together[0]):
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)


def test_question_rows_are_independent_under_reorder_and_sibling_change():
    model = build_tiny_model().eval()
    baseline = model.forward_one(encoded(branches=((4, 5), (6,))))
    changed_sibling = model.forward_one(encoded(branches=((4, 5), (30, 31, 32))))
    reordered = model.forward_one(encoded(branches=((6,), (4, 5))))
    torch.testing.assert_close(baseline[0], changed_sibling[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(baseline[0], reordered[1], atol=1e-6, rtol=1e-6)


def test_lora_and_head_gradients_leave_embedding_frozen():
    model = build_tiny_model()
    model.train()
    before = model.decoder.base_model.model.embed_tokens.weight.detach().clone()
    loss = sum(value.square().mean() for value in model.forward_one(encoded()))
    loss.backward()
    assert model.decoder.base_model.model.embed_tokens.weight.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters()
               if parameter.requires_grad and "lora_" in parameter.__class__.__name__.lower()) or any(
                   parameter.grad is not None for name, parameter in model.named_parameters() if "lora_" in name)
    assert model.head.query.weight.grad is not None
    torch.testing.assert_close(before, model.decoder.base_model.model.embed_tokens.weight)


def test_one_adam_step_changes_only_finite_lora_and_head_parameters():
    model = build_tiny_model().train()
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not parameter.requires_grad}
    logits = model.forward_one(encoded())
    loss = sum(torch.nn.functional.cross_entropy(value.unsqueeze(0), torch.tensor([label]))
               for value, label in zip(logits, (0, 0)))
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for name, parameter in model.named_parameters()
               if parameter.requires_grad and ("lora_B" in name or name.startswith("head.")))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if parameter.requires_grad}
    torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4).step()
    assert all(torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters() if name in frozen)
    assert any(not torch.equal(parameter, before[name]) for name, parameter in model.named_parameters()
               if name in before and ("lora_B" in name or name.startswith("head.")))


def test_tiny_architecture_has_all_seven_lora_targets_per_layer():
    model = build_tiny_model(layers=6)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" in name]
    assert all(any(f".{target}." in name for name in names) for target in
               ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    assert sum(name.count(".lora_A.") for name in names) == 42
    assert sum(name.count(".lora_B.") for name in names) == 42


def test_pointer_temperature_is_validated_and_probs_are_finite():
    model = build_tiny_model(temperature=2.0).eval()
    probabilities = model.probs(encoded())
    assert all(torch.isfinite(value).all() and torch.isclose(value.sum(), torch.tensor(1.0))
               for value in probabilities)


def test_temperature_is_applied_once_only_to_eval_logits():
    model = build_tiny_model(temperature=2.0)
    decision, options = torch.randn(64), torch.randn(2, 64)
    model.head.train()
    train_logits = model.head(decision, options)
    model.head.eval()
    eval_logits = model.head(decision, options)
    torch.testing.assert_close(eval_logits, train_logits / 2, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model.head.probabilities(eval_logits), torch.softmax(eval_logits, -1))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_actual_tiny_mps_matches_cpu_and_backpropagates():
    cpu = build_tiny_model().eval()
    mps = build_tiny_model().to("mps").eval()
    mps.load_state_dict(cpu.state_dict())
    cpu_probs, mps_probs = cpu.probs(encoded()), mps.probs(encoded())
    for expected, actual in zip(cpu_probs, mps_probs):
        torch.testing.assert_close(expected, actual.cpu(), atol=1e-3, rtol=1e-3)
    mps.train()
    sum(value.square().mean() for value in mps.forward_one(encoded())).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
               for parameter in mps.parameters())
