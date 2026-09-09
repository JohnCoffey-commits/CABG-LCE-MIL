"""Deterministic CABG development split and logical-block construction."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable, Mapping, Sequence

from reproduction.stage2i.cabg_math import CABGContractError, normalize_label
from reproduction.stage2i.constants import ABNORMAL_LABEL, NORMAL_LABEL
from reproduction.stage2i.fingerprint import canonical_json_sha256


SPLIT_DOMAIN = "cabg-v1.1-split"
ORDER_DOMAIN = "cabg-v1.1-order"
BLOCK_DOMAIN = "cabg-v1.1-block"
WITHIN_BLOCK_DOMAIN = "cabg-v1.1-within-block"


def _hash_text(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()


def _validated_records(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    required = {"sample_id", "sha256", "relative_path", "label"}
    rows = []
    identities = set()
    sample_ids = set()
    content_hashes = set()
    relative_paths = set()
    for source in records:
        if not required.issubset(source):
            raise CABGContractError(f"Sampler record missing keys: {sorted(required - set(source))}")
        row = dict(source)
        row["label"] = normalize_label(row["label"])
        identity = (str(row["sample_id"]), str(row["sha256"]), str(row["relative_path"]))
        if (
            identity in identities
            or identity[0] in sample_ids
            or identity[1] in content_hashes
            or identity[2] in relative_paths
        ):
            raise CABGContractError(f"Duplicate sampler identity component: {identity}")
        identities.add(identity)
        sample_ids.add(identity[0])
        content_hashes.add(identity[1])
        relative_paths.add(identity[2])
        rows.append(row)
    return rows


def split_development_pool(
    records: Iterable[Mapping[str, object]],
    *,
    expected_counts: Mapping[str, int] | None = None,
    validation_counts: Mapping[str, int] | None = None,
    mechanism_counts: Mapping[str, int] | None = None,
) -> dict[str, object]:
    rows = _validated_records(records)
    expected_counts = dict(expected_counts or {NORMAL_LABEL: 56, ABNORMAL_LABEL: 110})
    validation_counts = dict(validation_counts or {NORMAL_LABEL: 12, ABNORMAL_LABEL: 22})
    mechanism_counts = dict(mechanism_counts or {NORMAL_LABEL: 6, ABNORMAL_LABEL: 6})
    observed = Counter(row["label"] for row in rows)
    if dict(observed) != expected_counts:
        raise CABGContractError(
            f"Development-pool class counts changed: expected={expected_counts}, observed={dict(observed)}"
        )
    partitions = {"training-development": [], "threshold-validation": [], "mechanism-diagnostic": []}
    for label in (NORMAL_LABEL, ABNORMAL_LABEL):
        members = []
        for row in rows:
            if row["label"] != label:
                continue
            enriched = dict(row)
            enriched["split_key"] = _hash_text(
                SPLIT_DOMAIN, row["sha256"], row["relative_path"]
            )
            members.append(enriched)
        members.sort(key=lambda value: (value["split_key"], value["sample_id"]))
        validation_count = validation_counts[label]
        mechanism_count = mechanism_counts[label]
        if mechanism_count > validation_count or validation_count >= len(members):
            raise CABGContractError("Invalid validation/mechanism allocation.")
        validation = members[:validation_count]
        training = members[validation_count:]
        for index, row in enumerate(validation):
            row["cabg_split"] = "threshold-validation"
            row["cabg_split_index"] = index
        for index, row in enumerate(training):
            row["cabg_split"] = "training-development"
            row["cabg_split_index"] = index
        partitions["threshold-validation"].extend(validation)
        partitions["training-development"].extend(training)
        partitions["mechanism-diagnostic"].extend(dict(row) for row in validation[:mechanism_count])
    for values in partitions.values():
        values.sort(key=lambda value: (value["label"], value["split_key"], value["sample_id"]))
    train_counts = Counter(value["label"] for value in partitions["training-development"])
    validation_observed = Counter(value["label"] for value in partitions["threshold-validation"])
    mechanism_observed = Counter(value["label"] for value in partitions["mechanism-diagnostic"])
    return {
        **partitions,
        "counts": {
            "source": dict(observed),
            "training-development": dict(train_counts),
            "threshold-validation": dict(validation_observed),
            "mechanism-diagnostic": dict(mechanism_observed),
        },
        "split_fingerprint": canonical_json_sha256(partitions),
    }


class DeterministicBlockSampler:
    def __init__(self, training_records: Sequence[Mapping[str, object]], *, seed: int, epoch: int):
        self.records = _validated_records(training_records)
        self.seed = int(seed)
        self.epoch = int(epoch)
        counts = Counter(value["label"] for value in self.records)
        if counts != Counter({NORMAL_LABEL: 44, ABNORMAL_LABEL: 88}):
            raise CABGContractError(f"CABG training counts changed: {dict(counts)}")
        self._blocks = self._build()
        self.cursor = 0

    def _ordered_class(self, label: str) -> list[dict[str, object]]:
        members = [dict(value) for value in self.records if value["label"] == label]
        members.sort(
            key=lambda value: (
                _hash_text(ORDER_DOMAIN, self.seed, self.epoch, value["sha256"]),
                value["sample_id"],
            )
        )
        return members

    def _build(self) -> tuple[tuple[dict[str, object], ...], ...]:
        normal = self._ordered_class(NORMAL_LABEL)
        abnormal = self._ordered_class(ABNORMAL_LABEL)
        blocks = []
        for index, normal_record in enumerate(normal):
            members = [normal_record, abnormal[2 * index], abnormal[2 * index + 1]]
            members.sort(
                key=lambda value: (
                    _hash_text(
                        WITHIN_BLOCK_DOMAIN,
                        self.seed,
                        self.epoch,
                        index,
                        value["sha256"],
                    ),
                    value["sample_id"],
                )
            )
            block_key = _hash_text(
                BLOCK_DOMAIN,
                self.seed,
                self.epoch,
                *sorted(str(value["sha256"]) for value in members),
            )
            blocks.append((block_key, tuple(members)))
        blocks.sort(key=lambda value: value[0])
        return tuple(value[1] for value in blocks)

    @property
    def block_permutation_hash(self) -> str:
        payload = [[value["sample_id"] for value in block] for block in self._blocks]
        return canonical_json_sha256(payload)

    def state_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "block_permutation_hash": self.block_permutation_hash,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {"seed", "epoch", "cursor", "block_permutation_hash"}
        if set(state) != required:
            raise CABGContractError("Block-sampler state schema mismatch.")
        if int(state["seed"]) != self.seed or int(state["epoch"]) != self.epoch:
            raise CABGContractError("Block-sampler seed/epoch changed.")
        if str(state["block_permutation_hash"]) != self.block_permutation_hash:
            raise CABGContractError("Block permutation hash mismatch.")
        cursor = int(state["cursor"])
        if cursor < 0 or cursor > len(self._blocks):
            raise CABGContractError("Block-sampler cursor is invalid.")
        self.cursor = cursor

    def __iter__(self):
        while self.cursor < len(self._blocks):
            block = self._blocks[self.cursor]
            self.cursor += 1
            yield block

    def all_blocks(self) -> tuple[tuple[dict[str, object], ...], ...]:
        return self._blocks
