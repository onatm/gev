"""Pinned Kev suites: fetch, verify, sample, and training augmentation.

Split bytes are verified against SHA-256 hashes in the packaged manifests on
every load. Augmentation is Kev ``data.py`` at 08ab0b87d27cb5577a3b371ad7ed4e4686b0502b.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import urllib.request
from importlib import resources
from pathlib import Path

from .records import validate_request

HF_DATASET = "jaredpalmer/kev-suites"
HF_REVISION = "a88f56db5341397299137cb68775c2ea6e3f68cb"
SUITES = {
    "decision-v7": {"path": "v7/decision-v7",
                    "manifest_sha256": "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2"},
    "transfer-v4": {"path": "v4/transfer-v4",
                    "manifest_sha256": "31677c2256b406222e7d94ffdc0a02a70ce05746b9efe307876024c4e77291d1"},
}
SPLITS = ("train", "calibration", "development", "test")


class DataError(ValueError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(suite: str) -> dict:
    if suite not in SUITES:
        raise DataError(f"unknown suite: {suite}")
    data = resources.files("gev").joinpath(f"{suite}.manifest.json").read_bytes()
    if sha256(data) != SUITES[suite]["manifest_sha256"]:
        raise DataError(f"packaged manifest hash mismatch for {suite}")
    return json.loads(data)


def _file_info(files: dict, split: str) -> dict:
    info = files.get(f"{split}.jsonl") or files.get(split)
    if info is None:
        raise DataError(f"manifest has no {split} split")
    return info


def _verify(data: bytes, info: dict, name: str) -> list[dict]:
    if sha256(data) != info["sha256"]:
        raise DataError(f"hash mismatch for {name}")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    questions = sum(len(row.get("questions", {})) for row in rows)
    if (len(rows), questions) != (info["records"], info["questions"]):
        raise DataError(f"count mismatch for {name}: {len(rows)} records, {questions} questions")
    for row in rows:
        validate_request(row, labelled=True)
    return rows


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fetch(suite: str, split: str, data_root: str | Path = "data") -> Path:
    """Download one split from the pinned Hugging Face dataset revision and verify it."""
    info = _file_info(manifest(suite)["files"], split)
    url = (f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/"
           f"{SUITES[suite]['path']}/{split}.jsonl")
    with urllib.request.urlopen(url) as response:  # nosec B310: pinned HTTPS URL
        data = response.read()
    _verify(data, info, f"{suite}/{split}")
    path = Path(data_root) / suite / f"{split}.jsonl"
    _atomic_write(path, data)
    return path


def load_split(data_root: str | Path, suite: str, split: str) -> tuple[list[dict], str]:
    """Load verified rows and return them with the file's SHA-256.

    ``data_root`` may be the normal ``data`` layout (``<suite>/<split>.jsonl``) or
    a sample directory written by :func:`sample` (``<split>.jsonl`` + ``manifest.json``).
    """
    root = Path(data_root)
    if (root / "manifest.json").exists():
        local = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if local.get("parent", {}).get("suite") != suite:
            raise DataError(f"{root} is a sample of {local.get('parent', {}).get('suite')}, not {suite}")
        path, info = root / f"{split}.jsonl", _file_info(local["files"], split)
    else:
        path, info = root / suite / f"{split}.jsonl", _file_info(manifest(suite)["files"], split)
    if not path.exists():
        raise DataError(f"missing {path}; run `gev data fetch {suite} {split}`")
    data = path.read_bytes()
    return _verify(data, info, str(path)), sha256(data)


def sample(data_root: str | Path, out: str | Path, *, train_records: int = 128,
           dev_records: int = 64, seed: int = 0) -> dict:
    """Write a small, group-preserving, source-balanced decision-v7 subset for smoke runs."""
    out = Path(out)
    if out.exists():
        raise DataError(f"refusing to overwrite {out}")
    rng = random.Random(seed)

    def select(rows: list[dict], budget: int) -> list[dict]:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(row["_meta"].get("group_id", row["_meta"]["id"]), []).append(row)
        by_source: dict[str, list[list[dict]]] = {}
        for group in groups.values():
            by_source.setdefault(group[0]["_meta"].get("source", "unknown"), []).append(group)
        sources = list(by_source)
        rng.shuffle(sources)
        for source_groups in by_source.values():
            rng.shuffle(source_groups)
        selected, cursor = [], 0
        while sources and len(selected) < budget:
            source = sources[cursor % len(sources)]
            cursor += 1
            if not by_source[source]:
                sources.remove(source)
                continue
            group = by_source[source].pop(0)
            if not selected or len(selected) + len(group) <= budget:
                selected.extend(group)
        return selected

    files = {}
    for split, budget in (("train", train_records), ("development", dev_records)):
        rows = select(load_split(data_root, "decision-v7", split)[0], budget)
        data = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                        for row in rows)
        _atomic_write(out / f"{split}.jsonl", data)
        files[f"{split}.jsonl"] = {"sha256": sha256(data), "records": len(rows),
                                   "questions": sum(len(row["questions"]) for row in rows)}
    local = {"parent": {"suite": "decision-v7", "revision": HF_REVISION}, "seed": seed, "files": files}
    _atomic_write(out / "manifest.json", (json.dumps(local, indent=2) + "\n").encode())
    return local


# --- Training augmentation (Kev) ---------------------------------------------------------

NONE_OPTIONS = [
    ("other", "None of the above"), ("other", "A reason that fits none of the above"),
    ("none", "None of these"), ("other", "Something else"), ("not_listed", "Not listed here"),
    ("none_of_the_above", None), ("other", "A category that fits none of the above"),
    ("other", "None of the listed options apply"), ("unknown", "Cannot be determined from the options given"),
    ("other", "Other"), ("none", None), ("other", "An answer not covered by the other options"),
    ("no_match", "No option matches"),
]
DISTRACTORS = {"weather": "Bad weather caused it", "purple": "The colour purple",
               "pancakes": "A recipe for pancakes", "taxes": "Unrelated: quarterly tax filing"}


def item_rng(seed: int, epoch: int, identifier: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{epoch}:{identifier}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def augment(req: dict, rng: random.Random, p_none=.1, p_none_distract=.12, p_distract=.15) -> dict:
    """Kev's mutually exclusive choice augmentation: none-swap, none-add, or distractor."""
    if min(p_none, p_none_distract, p_distract) < 0 or p_none + p_none_distract + p_distract > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q
            continue
        crit, y = dict(q["criteria"]), q["label"]
        if q.get("target") is not None:
            keys = list(crit); rng.shuffle(keys)
            out["questions"][qid] = {**q, "criteria": {key: crit[key] for key in keys}}
            continue
        r = rng.random()
        none_options = [(key, value) for key, value in NONE_OPTIONS if key not in crit]
        distractors = [key for key in DISTRACTORS if key not in crit]
        if len(crit) > 2 and r < p_none and none_options:
            key, value = rng.choice(none_options); crit.pop(y); crit[key] = value; y = key
        elif p_none <= r < p_none + p_none_distract and len(crit) < 255 and none_options:
            key, value = rng.choice(none_options); crit[key] = value
        elif p_none + p_none_distract <= r < p_none + p_none_distract + p_distract and len(crit) < 255 and distractors:
            key = rng.choice(distractors); crit[key] = DISTRACTORS[key]
        keys = list(crit); rng.shuffle(keys)
        out["questions"][qid] = {**q, "criteria": {key: crit[key] for key in keys}, "label": y}
    return out


def none_pair(req: dict, rng: random.Random) -> list[dict]:
    """A contrastive pair: one choice question with a none-option present, then with the answer removed."""
    eligible = [(qid, q) for qid, q in req["questions"].items()
                if q["type"] == "choice" and len(q["criteria"]) >= 3]
    if not eligible:
        return []
    qid, q = rng.choice(eligible)
    choices = [(key, value) for key, value in NONE_OPTIONS if key not in q["criteria"]]
    key, value = rng.choice(choices or [("none_of_these", None)])
    keys = list(q["criteria"]) + [key]; rng.shuffle(keys)
    present = {**q, "criteria": {item: (value if item == key else q["criteria"][item]) for item in keys}}
    absent = {**present, "criteria": {item: val for item, val in present["criteria"].items()
                                      if item != q["label"]}, "label": key}
    return [{"state": req["state"], "questions": {qid: present}},
            {"state": req["state"], "questions": {qid: absent}}]


def request_id(request: dict) -> str:
    return request.get("_meta", {}).get("id", request.get("id", ""))


def training_variants(request: dict, *, seed: int, epoch: int, p_none: float, p_none_distract: float,
                      p_distract: float, p_none_pair: float) -> list[dict]:
    """The deterministic labelled requests one source record contributes in one epoch."""
    rng = item_rng(seed, epoch, request_id(request))
    variants = [augment(request, rng, p_none, p_none_distract, p_distract)]
    if rng.random() < p_none_pair:
        variants.extend(none_pair(request, rng))
    return variants


def epoch_order(requests: list[dict], seed: int, epoch: int) -> list[dict]:
    """Shuffle order for ``epoch``; replaying earlier shuffles makes resume exact and stateless."""
    rng, order = random.Random(seed), list(requests)
    for _ in range(epoch + 1):
        rng.shuffle(order)
    return order
