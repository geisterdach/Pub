#!/usr/bin/env python3
from typing import Dict, List, Any
from base import run_algorithm
import numpy as np
from scipy.optimize import linprog

coeffs = [1.596169194, 2.127659574, 16.077170418, 2.512562814]
# Define the bayesian matrix
M = np.array([
    [1.0,       0.385266, 0.804143, 0.236859],
    [0.289026,  1.0,      0.744974, 0.821357],
    [0.079927,  0.098701, 1.0,      0.105204],
    [0.150471,  0.695532, 0.67241,  1.0]
])


def solve_minimax(goal):
    n = M.shape[1]

    # Variables: v1, v2, v3, v4, t  (so total length = n+1 = 5)
    # Objective: minimize t -> coefficients [0,0,0,0,1]
    c = np.array([0, 0, 0, 0, 1])

    # Constraints:
    # 1) M v >= goal -> -M v <= -goal
    A_ub = np.hstack([-M, np.zeros((M.shape[0], 1))])
    b_ub = -goal

    # 2) v_i * coeff <= t
    #    (coeff * v_i) - t <= 0
    for i, coeff in enumerate(coeffs):
        row = np.zeros(n+1)
        row[i] = coeff
        row[-1] = -1
        A_ub = np.vstack([A_ub, row])
        b_ub = np.append(b_ub, 0.0)

    # Equality constraint: sum(v) = 1
    A_eq = np.zeros((1, n+1))
    A_eq[0, :n] = 1
    b_eq = np.array([1.0])

    # Bounds: v_i >= 0, t free but nonnegative (t >= 0)
    bounds = [(0, None)] * n + [(0, None)]

    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                  b_eq=b_eq, bounds=bounds, method="highs")
    return res


def algo_02_decision(
    constraints: List[Dict[str, Any]],
    attribute_statistics: Dict[str, Any],
    correlations: Dict[str, Dict[str, float]],
    admitted_count: int,
    rejected_count: int,
    next_person: Dict[str, Any],
    accepted_count: Dict[str, int]
) -> bool:
    remaining_spots = 1000 - admitted_count

    # Calculate values for each constraint attribute
    values = {}
    for constraint in constraints:
        attr_id = constraint["attribute"]
        min_count = constraint["minCount"]
        current_accepted = accepted_count.get(attr_id, 0)
        remaining_needed = min_count - current_accepted

        if remaining_needed <= 0:
            values[attr_id] = 0.0
        else:
            values[attr_id] = remaining_needed / remaining_spots

        # define the vector v
    goal = np.array([values.get("techno_lover", 0.0), values.get("well_connected", 0.0),
                     values.get("creative", 0.0), values.get("berlin_local", 0.0)])

    # To make it solvable, we let well_connected be always 0.0
    goal = np.array([values.get("techno_lover", 0.0), 0.0,
                     values.get("creative", 0.0), values.get("berlin_local", 0.0)])

    res = solve_minimax(goal)
    if res.success:
        v = res.x[:-1]
        t = res.x[-1]
        acceptance_rates = coeffs / t * v
        print("")
        print("Goal:     ", goal)
        print("(Remaing needed):", {k: constraint["minCount"] - accepted_count.get(k, 0)
              for constraint in constraints for k in [constraint["attribute"]]})
        print("Remaining spots:", remaining_spots)
        print("Optimal v:", v)
        print("Optimal t:", t)
        print("Mv:       ", M @ v)
        print("Accept r: ", acceptance_rates)
        # Prognose is as follows:
        # take remaing spots and multiply with t and substract t, then add already admitted
        prognose = rejected_count + remaining_spots * (t - 1)
        print("Prognose admitted after all:", prognose)
        # We categorize the attributes (acceptance_rate > 0.99 urgently needed, acceptance_rate = 0 irrelevant, acceptance_rate in between normally needed)
        category = {}
        for i, attr_id in enumerate(["techno_lover", "well_connected", "creative", "berlin_local"]):
            if acceptance_rates[i] > 0.99:
                category[attr_id] = "urgent"
            elif acceptance_rates[i] == 0:
                category[attr_id] = "irrelevant"
            else:
                category[attr_id] = "normal"
        # print("Categories:", category)
        # Now we decide based on the categories
        person_attributes = next_person.get("attributes", {})
        # if the person has any urgent attribute, accept
        for attr_id, cat in category.items():
            if cat == "urgent" and person_attributes.get(attr_id, False):
                print(f"Accepting because of urgent attribute: {attr_id}")
                return True
        # if the person has every normal attribute (and normal attributes are not empty), accept
        normal_attrs = [attr_id for attr_id,
                        cat in category.items() if cat == "normal"]
        if all(person_attributes.get(attr_id, False) for attr_id in normal_attrs) and normal_attrs:
            print(
                f"Accepting because has all normal attributes: {normal_attrs}")
            return True
        # else reject
        # print("Rejecting the person")
        return False

    else:
        print("")
        print("Goal:     ", goal)
        print("(Remaing needed):", {k: constraint["minCount"] - accepted_count.get(k, 0)
              for constraint in constraints for k in [constraint["attribute"]]})
        print("Remaining spots:", remaining_spots)
        # This happens, only in some very rare cases
        # In this case, we say an attribute is normal, if remaining_needed = remaining_spots
        category = {}
        for constraint in constraints:
            attr_id = constraint["attribute"]
            min_count = constraint["minCount"]
            current_accepted = accepted_count.get(attr_id, 0)
            remaining_needed = min_count - current_accepted

            if remaining_needed == remaining_spots:
                category[attr_id] = "normal"
            else:
                category[attr_id] = "irrelevant"
        # We know accept if the person has all normal attributes
        person_attributes = next_person.get("attributes", {})
        normal_attrs = [attr_id for attr_id,
                        cat in category.items() if cat == "normal"]
        if all(person_attributes.get(attr_id, False) for attr_id in normal_attrs):
            print(
                f"Accepting because has all normal attributes: {normal_attrs}")
            return True
        # else reject
        # print("Rejecting the person")
        return False


if __name__ == "__main__":
    run_algorithm(algo_02_decision, scenario=2, algo_name="algo_02")
