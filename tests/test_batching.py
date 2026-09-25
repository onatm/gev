from gev.training.batching import physical_token_count


def _encoded(questions):
    state = [6, 3]
    ids, segments, positions, decide, options = list(state), [0, 0], [0, 1], [], []
    for index, branch in enumerate(questions, start=1):
        start = len(ids)
        ids.extend(branch)
        segments.extend([index] * len(branch))
        positions.extend(range(len(state), len(state) + len(branch)))
        decide.append(len(ids) - 1)
        options.append([start + 2, start + 4])
    return {"ids": ids, "seg": segments, "pos": positions,
            "decide_idx": decide, "opt_idx": options, "state_length": len(state)}


def test_row_physical_tokens_repeat_state_per_question_and_support_zero_questions():
    encoded = _encoded([[7, 1, 8, 3, 9, 10], [7, 2, 8, 4, 9, 10, 5]])
    assert physical_token_count(encoded) == (2 + 6) + (2 + 7)
    assert physical_token_count(_encoded([])) == 0
