#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    data_root = Path(
        os.environ.get(
            "MEDIC_AD_VQARAD_TINY_ROOT",
            "/home/data/medic-ad/training-tiny-vqarad",
        )
    )
    source = data_root / "validation.json"
    target = data_root / "validation-diagnostic-first.json"
    records = json.loads(source.read_text(encoding="utf-8"))
    if len(records) != 4:
        raise RuntimeError(f"Expected four validation records, got {len(records)}")
    selected = records[:1]
    encoded = (json.dumps(selected, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if target.exists() and target.read_bytes() != encoded:
        raise RuntimeError(f"Refusing to overwrite mismatched diagnostic annotation: {target}")
    target.write_bytes(encoded)
    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "source_records": len(records),
                "diagnostic_records": len(selected),
                "selected_id": selected[0]["id"],
                "annotation_sha256": hashlib.sha256(encoded).hexdigest(),
                "output": str(target),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
