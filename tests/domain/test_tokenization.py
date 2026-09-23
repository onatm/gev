from types import SimpleNamespace

import pytest

from gev.domain.tokenization import MarkerError, MarkerMap, encode, pad_rows, rows_of


class Tok:
    all_special_tokens = ["<ctrl>"]
    bos_token_id, eos_token_id, pad_token_id, unk_token_id = 90, 91, 92, 93
    def __init__(self):
        self.vocab = {f"<m{i}>": 10 + i for i in range(5)}
    def get_vocab(self): return self.vocab
    def __call__(self, text, add_special_tokens=False):
        if text in self.vocab: return SimpleNamespace(input_ids=[self.vocab[text]])
        return SimpleNamespace(input_ids=[1] * len(text))


def markers(tok):
    return MarkerMap.from_dict({"tokenizer_revision": "rev", "bos": False, "roles": {
        role: {"string": f"<m{i}>", "id": 10 + i} for i, role in enumerate(("state", "question", "option_start", "option_end", "decide"))}}, tok)


def test_positions_rows_and_padding():
    tok = Tok(); mm = markers(tok)
    rec = {"state": "s", "questions": [{"instr": "i", "options": ["a", "b"], "label": 1}, {"instr": "j", "options": ["c"], "label": 0}]}
    enc = encode(tok, rec, mm, state_cap=10, branch_cap=20, packed_cap=50)
    assert enc["pos"][enc["state_length"]] == enc["state_length"]
    state, state_pos, rows = rows_of(enc)
    assert rows[0]["pos"][0] == len(state) and rows[1]["pos"][0] == len(state)
    padded = pad_rows(rows, 99)
    assert padded["attention"][1][-1] == 0


def test_marker_user_text_and_caps_are_not_structural():
    tok = Tok(); mm = markers(tok)
    rec = {"state": "<m0> <|x|>", "questions": [{"instr": "i", "options": ["a"], "label": 0}]}
    enc = encode(tok, rec, mm, state_cap=30, branch_cap=20, packed_cap=50)
    assert enc["ids"].count(mm.ids["state"]) == 1
    with pytest.raises(MarkerError): encode(tok, rec, mm, state_cap=1)
