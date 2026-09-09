#!/usr/bin/env python3

from reproduction.stage2f.gradient_evidence import has_finite_nonzero_element


def main() -> None:
    underflowed_norm = {
        "gradient_present": True,
        "finite": True,
        "nonzero": False,
        "norm": 0.0,
        "max_abs": 1.3252863207712998e-28,
    }
    assert has_finite_nonzero_element(underflowed_norm)
    assert not has_finite_nonzero_element({**underflowed_norm, "max_abs": 0.0})
    assert not has_finite_nonzero_element({**underflowed_norm, "finite": False})
    print("gradient-evidence max-abs criterion: PASS")


if __name__ == "__main__":
    main()
