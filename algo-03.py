from itertools import combinations
import numpy as np
from scipy.optimize import linprog
from typing import List, Dict, Any
from base import run_algorithm
# Empirical data from analyse-data.py
data = {(): 0.013271228207549342, (0,): 0.006759402132608646, (1,): 0.01634577738882108, (2,): 0.012925773243361508, (3,): 0.00023030330945855693, (4,): 0.000224545726722093, (5,): 0.022149420787176712, (0, 1): 0.042542778839731926, (0, 2): 0.01304668248082725, (0, 3): 0.000598788604592248, (0, 4): 0.0014796987632712281, (0, 5): 0.15827019184265678, (1, 2): 0.15216715414200502, (1, 3): 0.0004951521153358974, (1, 4): 0.00019000023030330945, (1, 5): 0.006246977269063357, (2, 3): 0.0008866677414154442, (2, 4): 8.636374104695885e-05, (2, 5): 0.0637249257271827, (3, 4): 0.0002475760576679487, (3, 5): 0.0002648488058773405, (4, 5): 0.0007139402593215264, (0, 1, 2): 0.26124455908431404, (0, 1, 3): 0.001122728633610465, (0, 1, 4): 0.00120333479192096, (0, 1, 5): 0.013466986020589116, (0, 2, 3): 0.0018712143893507749, (0, 2, 4): 0.0004778793671265056, (0, 2, 5): 0.10466709656617766, (0, 3, 4): 0.0010996983026646094, (0, 3, 5): 0.0016063655834734345, (0, 4, 5): 0.006684553557034615, (1, 2, 3): 0.0019000023030330945, (1, 2, 4): 0.000529697611754681, (1, 2, 5): 0.020923055664309895,
        (1, 3, 4): 0.00032242463324197966, (1, 3, 5): 0.00030515188503258794, (1, 4, 5): 0.0004951521153358974, (2, 3, 4): 0.0005642431081734644, (2, 3, 5): 0.0010248497270905783, (2, 4, 5): 0.00031666705050551575, (3, 4, 5): 0.0004548490361806499, (0, 1, 2, 3): 0.004393035627921973, (0, 1, 2, 4): 0.0012206075401303518, (0, 1, 2, 5): 0.024717302687639622, (0, 1, 3, 4): 0.0012263651228668156, (0, 1, 3, 5): 0.001059395223509362, (0, 1, 4, 5): 0.004370005296976118, (0, 2, 3, 4): 0.00179060823104028, (0, 2, 3, 5): 0.002665760806982796, (0, 2, 4, 5): 0.0019000023030330945, (0, 3, 4, 5): 0.0028730337854954977, (1, 2, 3, 4): 0.0009384859860436194, (1, 2, 3, 5): 0.0011918196264480321, (1, 2, 4, 5): 0.00020151539577623732, (1, 3, 4, 5): 0.0004433338707077221, (2, 3, 4, 5): 0.0006448492664839593, (0, 1, 2, 3, 4): 0.002706063886138044, (0, 1, 2, 3, 5): 0.003345155569885539, (0, 1, 2, 4, 5): 0.0014106077704336612, (0, 1, 3, 4, 5): 0.0027981852099214664, (0, 2, 3, 4, 5): 0.002233942101748002, (1, 2, 3, 4, 5): 0.001047880058036434, (0, 1, 2, 3, 4, 5): 0.003673337785863983}


# Features
n_features = 6

# ---- ALL SUBSETS ----
all_subsets = []
for r in range(0, n_features + 1):  # skip empty set
    all_subsets.extend(combinations(range(n_features), r))
n_subsets = len(all_subsets)
print(f"Number of subsets: {n_subsets}")

# Matrix that maps subsets -> features
M_subsets = np.zeros((n_features, n_subsets))
for idx, subset in enumerate(all_subsets):
    for i in subset:
        # We add some epsilon from 0.00001 for subsets of size 1, 0.000001 for size 2, etc
        M_subsets[i, idx] += 1
        M_subsets[i, idx] += 1e-5 / pow(10, len(subset)-1)

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
        row[i] = 1/coeff
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


def algo_03_decision(
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
    goal = np.array([values.get("underground_veteran", 0.0), values.get("international", 0.0),
                     values.get("fashion_forward", 0.0), values.get(
                         "queer_friendly", 0.0),
                     values.get("vinyl_collector", 0.0), values.get("german_speaker", 0.0)])
    # To make it solvable, we let underground_veteran and fashion_forward be 0
    goal = np.array([0.0, goal[1], 0.0, goal[3], goal[4], goal[5]])
    res = solve_minimax(goal)

    if res.success:
        v = res.x[:-1]
        t = res.x[-1]

        acceptance_rates = 1/coeffs_subsets / t * v

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

        # An indices combination with 1/coeff * v_subset >= t - epsilon is called limiting factor
        # Print all limiting factors
        # print("\nLimiting factors (coeff * v_subset >= t - epsilon):")
        # for idx, coeff in enumerate(coeffs_subsets):
        #     if 1/coeff * v[idx] >= t - 1e-6:
        #         print(
        #             f"Subset {all_subsets[idx]}: coeff = {coeff:.6f}, v = {v[idx]:.6f}")

        # For a person define the property Matrix which is 6x6 with 1 if person has both attributes i and j
        person_attributes = next_person.get("attributes", {})
        # We iterate over all subsets with acceptance rate > 0, and check if the person has all attributes in the subset,
        # Note that subsets are numbers but person_attributes is an dict with string keys like "underground_veteran"
        # Map attributes to indices
        attr_names = ["underground_veteran", "international", "fashion_forward",
                      "queer_friendly", "vinyl_collector", "german_speaker"]

        # Set of indices for attributes the person actually has
        person_indices = {i for i, name in enumerate(
            attr_names) if person_attributes.get(name, False)}

        # Now check if subset is fully contained in person_indices
        for idx, subset in enumerate(all_subsets):
            if acceptance_rates[idx] > 0.33:
                if set(subset).issubset(person_indices):
                    print(
                        f"Person accepted because of {subset} beeing a subset of {person_indices}")
                    return True

        return False

    else:
        print("No feasible solution found.")
        print("Goal:", goal)
        return True


if __name__ == "__main__":
    run_algorithm(algo_03_decision, scenario=3, algo_name="algo_03")
