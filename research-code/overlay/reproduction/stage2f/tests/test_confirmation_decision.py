#!/usr/bin/env python3

from reproduction.stage2f.verify_confirmation_pair import confirmation_decision


def main() -> None:
    _, outcome, decision = confirmation_decision(
        b0_reliability_failure=False,
        a3_reliability_failure=True,
        overall_em_delta=0.0,
        eval_loss_delta=0.0,
    )
    assert outcome == "METHOD_SPECIFIC_DEGRADATION_REPRODUCED"
    assert decision == "PAUSE_FB_MAQ"

    stable, _, decision = confirmation_decision(
        b0_reliability_failure=False,
        a3_reliability_failure=False,
        overall_em_delta=-0.125,
        eval_loss_delta=0.5,
    )
    assert stable and decision == "PAUSE_FB_MAQ"

    _, outcome, decision = confirmation_decision(
        b0_reliability_failure=False,
        a3_reliability_failure=False,
        overall_em_delta=0.0,
        eval_loss_delta=0.0,
    )
    assert outcome == "COLLAPSE_NOT_REPRODUCED_INSTABILITY_REMAINS"
    assert decision == "DRAFT_STAGE2G_PILOT_PROTOCOL"

    _, outcome, decision = confirmation_decision(
        b0_reliability_failure=True,
        a3_reliability_failure=True,
        overall_em_delta=-1.0,
        eval_loss_delta=10.0,
    )
    assert outcome == "COMMON_RUN_FAILURE"
    assert decision == "BLOCKED_COMMON_FAILURE"
    print("confirmation decision rule: PASS")


if __name__ == "__main__":
    main()
