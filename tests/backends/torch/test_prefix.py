import pytest
import torch

from gev.backends.torch.gemma3 import build_tiny_model
from gev.domain.tokenization import rows_of


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


def test_prefix_boundary_short_and_long_state_answers():
    model = build_tiny_model().eval()
    value = encoded(state=tuple(range(7)))
    state, positions, questions = rows_of(value)
    prefix = model.prefill_prefix(state, positions)
    result = model.forward_prefix_batch(prefix, questions)
    assert len(result) == len(questions)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("state_length", [7, 8, 9])
def test_prefix_logits_match_rows_for_every_question_and_repeated_use(implementation, state_length):
    model = build_tiny_model().eval()
    model.backbone.config._attn_implementation = implementation
    value = encoded(state=tuple(range(state_length)), branches=((4, 5, 6, 7, 8), (6,)))
    state, positions, questions = rows_of(value)
    prefix = model.prefill_prefix(state, positions)
    expected = model.forward_one(value)
    first = model.forward_with_prefix(value, prefix)
    second = model.forward_with_prefix(value, prefix)
    for left, right, again in zip(expected, first, second):
        torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(right, again, atol=1e-6, rtol=1e-6)


def test_prefix_cache_is_unchanged_after_two_answers():
    model = build_tiny_model().eval()
    value = encoded()
    state, positions, questions = rows_of(value)
    prefix = model.prefill_prefix(state, positions)
    before = prefix.cache.get_seq_length()
    model.forward_prefix_batch(prefix, questions)
    model.forward_prefix_batch(prefix, list(reversed(questions)))
    assert prefix.cache.get_seq_length() == before


def test_prefix_rejects_other_model_and_training_mode():
    value = encoded()
    first = build_tiny_model().eval()
    state, positions, questions = rows_of(value)
    prefix = first.prefill_prefix(state, positions)
    with pytest.raises(ValueError):
        build_tiny_model().eval().forward_prefix_batch(prefix, questions)
    first.train()
    with pytest.raises(RuntimeError):
        first.forward_prefix_batch(prefix, questions)


def test_prefix_rejects_stale_parameter_versions():
    model = build_tiny_model().eval()
    value = encoded()
    state, positions, questions = rows_of(value)
    prefix = model.prefill_prefix(state, positions)
    with torch.no_grad():
        model.head.query.weight.add_(1e-4)
    with pytest.raises(ValueError):
        model.forward_prefix_batch(prefix, questions)


def test_prefix_rejects_same_length_wrong_state():
    model = build_tiny_model().eval()
    original = encoded(state=(2, 3))
    other = encoded(state=(7, 8))
    state, positions, _ = rows_of(original)
    prefix = model.prefill_prefix(state, positions)
    with pytest.raises(ValueError):
        model.forward_with_prefix(other, prefix)
