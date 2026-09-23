import torch

from gev.backends.torch.gemma3 import build_tiny_model
from gev.backends.torch.masks import packed_attention_mask


def encoded(state=(2, 3), branches=((4, 5), (6,))):
    ids, pos, seg, opt = list(state), list(range(len(state))), [0] * len(state), [-1] * len(state)
    decides, ends = [], []
    for question, options in enumerate(branches, 1):
        start, branch_position = len(ids), len(state)
        for option in options:
            ids.extend((10, option)); pos.extend((branch_position, branch_position + 1)); branch_position += 2
            seg.extend((question, question)); opt.extend((len(ends), len(ends)))
        ids.append(20); pos.append(branch_position); seg.append(question); opt.append(-2)
        ends.append([start + 1, start + 3][:len(options)]); decides.append(len(ids) - 1)
    return {"ids": ids, "pos": pos, "seg": seg, "opt": opt, "decide_idx": decides,
            "opt_idx": ends, "labels": [0] * len(branches), "state_length": len(state)}


def test_full_mask_is_causal_and_blocks_sibling_and_future_question():
    seg = torch.tensor([[0, 0, 1, 1, 2, 2]])
    pos = torch.tensor([[0, 1, 2, 3, 2, 3]])
    mask = packed_attention_mask(seg, pos, "full_attention")[0, 0]
    assert mask[2, 0] and mask[2, 2]
    assert not mask[2, 4] and not mask[4, 2]
    assert not mask[2, 3]  # causal, not future within the question


def test_sliding_window_uses_logical_not_physical_distance():
    seg = torch.tensor([[0, 0, 1, 1]])
    pos = torch.tensor([[0, 511, 512, 513]])
    mask = packed_attention_mask(seg, pos, "sliding_attention", window=2)[0, 0]
    assert mask[2, 1] and not mask[2, 0]


def test_padding_has_finite_query_diagonal_and_no_real_key_visibility():
    seg = torch.tensor([[0, 1, -1]])
    pos = torch.tensor([[0, 1, 0]])
    valid = torch.tensor([[True, True, False]])
    mask = packed_attention_mask(seg, pos, "full_attention", valid=valid, dtype=torch.float32)[0, 0]
    assert torch.isfinite(mask[2, 2]) and mask[2, 0] == torch.finfo(torch.float32).min
    assert mask[0, 2] == torch.finfo(torch.float32).min


def test_packed_batch_preserves_record_and_question_order():
    model = build_tiny_model().eval()
    first, second = encoded(), encoded(state=(7, 8, 9), branches=((4,),))
    expected = [model.forward_one(value) for value in (first, second)]
    actual = model.forward_packed_batch([first, second])
    for left_record, right_record in zip(expected, actual):
        for left, right in zip(left_record, right_record):
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)


def test_logical_gap_does_not_leak_a_changed_sibling():
    model = build_tiny_model().eval()
    baseline = model.forward_packed_one(encoded(branches=((4, 5), (6,))))[0]
    changed = model.forward_packed_one(encoded(branches=((4, 5), (30, 31, 32))))[0]
    torch.testing.assert_close(baseline, changed, atol=1e-5, rtol=1e-5)


def test_rows_and_packed_share_lora_and_pointer_gradients_without_dropout():
    row_model, packed_model = build_tiny_model().train(), build_tiny_model().train()
    for model in (row_model, packed_model):
        for module in model.modules():
            if hasattr(module, "lora_dropout"):
                for dropout in module.lora_dropout.values():
                    dropout.p = 0.0
    value = encoded()
    row_loss = sum(item.square().mean() for item in row_model.forward_rows_batch([value])[0])
    packed_loss = sum(item.square().mean() for item in packed_model.forward_packed_batch([value])[0])
    row_loss.backward(); packed_loss.backward()
    row_grads = {name: parameter.grad for name, parameter in row_model.named_parameters() if parameter.grad is not None}
    packed_grads = {name: parameter.grad for name, parameter in packed_model.named_parameters() if parameter.grad is not None}
    assert row_grads.keys() == packed_grads.keys()
    for name in row_grads:
        torch.testing.assert_close(row_grads[name], packed_grads[name], atol=2e-4, rtol=2e-4)
