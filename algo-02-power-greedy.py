from itertools import combinations
import numpy as np
from scipy.optimize import linprog
from typing import List, Dict, Any
from base import run_algorithm

# ---- Empirical data (Scenario 2) ----
# Hier musst du die Daten aus analyse-data.py für Scenario 2 einsetzen
data = {(): 0.032898891292870526, (0,): 0.4199577635379248, (1,): 0.033209453514011535, (2,): 0.0011594322922597543, (3,): 0.04844253046097786, (0, 1): 0.09591714199939957, (0, 2): 0.005833393720431889, (0, 3): 0.013809666766736716, (1, 2)
         : 0.0025983705835464135, (1, 3): 0.2474404496940962, (2, 3): 0.0017184442903135644, (0, 1, 2): 0.010797213221668961, (0, 1, 3): 0.047034648391805295, (0, 2, 3): 0.006765080383854905, (1, 2, 3): 0.00647004627377095, (0, 1, 2, 3): 0.025947473576331016}

# Features
n_features = 4
feature_names = ["techno_lover", "well_connected", "creative", "berlin_local"]

# ---- ALL SUBSETS ----
all_subsets = []
for r in range(0, n_features + 1):
    all_subsets.extend(combinations(range(n_features), r))
n_subsets = len(all_subsets)
print(f"Number of subsets: {n_subsets}")

# Matrix that maps subsets -> features
M_subsets = np.zeros((n_features, n_subsets))
for idx, subset in enumerate(all_subsets):
    for i in subset:
        # epsilon gegen degenerierte Lösungen
        M_subsets[i, idx] += 1
        M_subsets[i, idx] += 1e-5 / pow(10, len(subset) - 1)

# Coefficients from empirical data
coeffs_subsets = np.array([data.get(subset, 0.0) for subset in all_subsets])


def solve_minimax(goal):
    c = np.zeros(n_subsets + 1)
    c[-1] = 1  # minimize t

    # 1) Goal constraints: M*v >= goal
    A_ub = np.hstack([-M_subsets, np.zeros((n_features, 1))])
    b_ub = -goal

    # 2) 1/coeff * v_subset <= t
    A_rows = []
    b_vals = []
    for i, coeff in enumerate(coeffs_subsets):
        row = np.zeros(n_subsets + 1)
        if coeff > 0:  # nur wo Daten > 0
            row[i] = 1 / coeff
            row[-1] = -1
            A_rows.append(row)
            b_vals.append(0.0)

    if A_rows:
        A_ub = np.vstack([A_ub] + A_rows)
        b_ub = np.concatenate([b_ub, b_vals])

    # 3) Normalization: sum v_subset = 1
    A_eq = np.zeros((1, n_subsets + 1))
    A_eq[0, :n_subsets] = 1
    b_eq = np.array([1.0])

    bounds = [(0, None)] * (n_subsets + 1)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
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

    # Goal vector für die 4 Features
    goal = np.array([values.get(name, 0.0) for name in feature_names])

    res = solve_minimax(goal)

    if res.success:
        v = res.x[:-1]
        t = res.x[-1]

        acceptance_rates = 1 / coeffs_subsets / t * v

        print("\nSample of nonzero v-subsets:")
        for idx, subset in enumerate(all_subsets):
            if v[idx] > 1e-6:
                print(
                    f"Subset {subset}: v = {v[idx]:.3f}, A = {acceptance_rates[idx]:.3f}")

        print("\nOptimal t (minimized max):", np.round(t, 3))
        print("M_subsets @ v:", np.round(M_subsets @ v, 3))
        print("Goal:", np.round(goal, 3))

        prognose = rejected_count + remaining_spots * (t - 1)
        print("Prognose admitted after all:", np.round(prognose, 1))

        # Check person attributes
        person_attributes = next_person.get("attributes", {})
        print("Person attributes: ", person_attributes)
        # Set of indices for attributes the person actually has
        person_indices = {i for i, name in enumerate(
            feature_names) if person_attributes.get(name, False)}
        acceptance_sum = 0.0
        # Now check if subset is fully contained in person_indices
        for idx, subset in enumerate(all_subsets):
            if set(subset).issubset(person_indices):
                acceptance_sum += acceptance_rates[idx]

        print("acceptance_sum", np.round(acceptance_sum, 1))
        return acceptance_sum > 0.1

    else:
        print("No feasible solution found.")
        print("Goal:", goal)
        return True


if __name__ == "__main__":
    run_algorithm(algo_02_decision, scenario=2, algo_name="algo_02")
