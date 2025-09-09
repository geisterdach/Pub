#!/usr/bin/env python3
from typing import Dict, List, Any
from base import run_algorithm


def algo_01_decision(
    constraints: List[Dict[str, Any]],
    attribute_statistics: Dict[str, Any],
    correlations: Dict[str, Dict[str, float]],
    admitted_count: int,
    rejected_count: int,
    next_person: Dict[str, Any],
    accepted_count: Dict[str, int]
) -> bool:
    # Distribution of attributes
    ###########
    # 32,25% young
    # 32,25% dressed
    ###########
    # 14,398325% young &  dressed
    # 17,851675% young & not dressed
    # 17,851675% not young & dressed
    # 49,9000% not young & not dressed
    ###########
    # Define some variables
    p1x = 0.3225  # P(young)
    px1 = 0.3225  # P(dressed)
    p11 = 0.14398325  # P(young & dressed)
    p10 = 0.17851675  # P(young & not dressed)
    p01 = 0.17851675  # P(not young & dressed)
    p00 = 0.49900  # P(not young & not dressed)
    # Def constants
    young_min = 600
    well_dressed_min = 600
    count = 1000
    admitted_young = accepted_count.get("young", 0)
    admitted_dressed = accepted_count.get("well_dressed", 0)
    constrained_count = count - admitted_count
    constrained_young = max(young_min - admitted_young, 0)
    constrained_dressed = max(well_dressed_min - admitted_dressed, 0)
    # Compute optimal distribution
    # Check whats the next person belongs to
    person_attributes = next_person.get("attributes", {})
    is_young = person_attributes.get("young", False)
    is_dressed = person_attributes.get("well_dressed", False)

    # Define optimal ratios
    r10 = 1 - 1.0 * constrained_dressed / constrained_count
    r01 = 1 - 1.0 * constrained_young / constrained_count
    r11 = 1 - r10 - r01
    print(
        f"Constrained counts: young={constrained_young}, dressed={constrained_dressed}, total={constrained_count}")
    print(f"Optimal ratios: r10={r10}, r01={r01}, r11={r11}")
    print(f"Person attributes: is_young={is_young}, is_dressed={is_dressed}")

    # Decision
    # Corner cases
    if r11 < -0.000001:
        # In this case we can accept everyone without ANY risk
        return True
    if r11 < p11:
        # In this case we can accept anyone that makes any contribution
        # We could be a bit more greedy, but we dont
        if is_young and r10 > 0.000001:
            return True
        if is_dressed and r01 > 0.000001:
            return True
        if is_young and is_dressed and r11 > 0.000001:  # The ladder is always true if both first are true
            return True
        return False
    # Else r11 >= p11
    # We are using the same strategy here without any choice.
    if is_young and r10 > 0.000001:
        return True
    if is_dressed and r01 > 0.000001:
        return True
    if is_young and is_dressed and r11 > 0.000001:  # The ladder is always true if both first are true
        return True
    return False


if __name__ == "__main__":

    run_algorithm(algo_01_decision, scenario=1, algo_name="algo_01")
