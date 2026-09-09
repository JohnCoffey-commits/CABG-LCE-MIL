#!/usr/bin/env python3

from types import SimpleNamespace

from qwenvl.train.train_qwen import TargetGradientAuditCallback


def make_callback(*, nonfinite: bool = False):
    callback = TargetGradientAuditCallback(interval=16, strict=True)
    callback._expected = {"target.weight": 4}
    callback._seen_gradient = {"target.weight": True}
    callback._seen_finite = {"target.weight": True}
    callback._seen_nonfinite = {"target.weight": nonfinite}
    callback._seen_nonzero = {"target.weight": False}
    records = []
    callback._write_jsonl = records.append
    return callback, records


def main() -> None:
    state = SimpleNamespace(global_step=32)

    sparse_zero, sparse_zero_records = make_callback()
    sparse_zero.on_train_end(None, state, None)
    sparse_zero_summary = sparse_zero_records[-1]
    assert sparse_zero_summary["status"] == "SUCCESS"
    assert sparse_zero_summary["sampled_activity_status"] == "SPARSE_ZERO_OBSERVED"
    assert sparse_zero_summary["never_nonzero"] == ["target.weight"]

    nonfinite, _ = make_callback(nonfinite=True)
    try:
        nonfinite.on_train_end(None, state, None)
    except RuntimeError as exc:
        assert "Strict trainable-gradient audit failed" in str(exc)
    else:
        raise AssertionError("Strict audit must reject a non-finite sampled gradient.")

    print("gradient-audit semantics: PASS")


if __name__ == "__main__":
    main()
