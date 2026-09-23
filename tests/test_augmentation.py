import random
from fixtures.kev_reference import augment as reference_augment

from gev.data.augmentation import augment, none_pair


def request():
    return {"state": "evidence", "questions": {
        "q": {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}, "label": "a"},
        "soft": {"type": "choice", "criteria": {"x": "X", "y": "Y"}, "label": "x", "target": {"x": .25, "y": .75}},
    }}


def test_forced_none_and_distractor_are_exclusive():
    none = augment(request(), random.Random(1), p_none=1, p_none_distract=0, p_distract=0)
    assert len(none["questions"]["q"]["criteria"]) == 3
    distract = augment(request(), random.Random(1), p_none=0, p_none_distract=0, p_distract=1)
    assert len(distract["questions"]["q"]["criteria"]) == 4


def test_none_pair_is_one_question_and_labels_present_absent():
    pair = none_pair(request(), random.Random(2))
    assert len(pair) == 2 and list(pair[0]["questions"]) == ["q"]
    present, absent = pair
    assert present["state"] == absent["state"]
    assert len(present["questions"]["q"]["criteria"]) == 4
    assert len(absent["questions"]["q"]["criteria"]) == 3
    assert absent["questions"]["q"]["label"] in absent["questions"]["q"]["criteria"]


def test_soft_target_is_permuted_by_key_without_losing_mass():
    out = augment(request(), random.Random(3), p_none=1, p_none_distract=0, p_distract=0)
    assert set(out["questions"]["soft"]["criteria"]) == {"x", "y"}
    assert out["questions"]["soft"]["target"] == {"x": .25, "y": .75}


def test_matches_pinned_reference_for_many_seeds():
    for seed in range(40):
        assert augment(request(), random.Random(seed), .1, .12, .15) == reference_augment(request(), random.Random(seed), .1, .12, .15)
