from itertools import combinations_with_replacement, combinations
import numpy as np
from scipy.optimize import linprog
from typing import List, Dict, Any
from base import run_algorithm

# Feature frequencies
relativeFrequencies = np.array([
    0.679500,  # underground_veteran
    0.573500,  # international
    0.691000,  # fashion_forward
    0.046140,  # queer_friendly
    0.044540,  # vinyl_collector
    0.456500   # german_speaker
])

# Feature correlation matrix
M = np.array([
    [1.000000, 0.646861, 0.626556, 0.758431, 0.835631, 0.736473],
    [0.545953, 1.000000, 0.697757, 0.581751, 0.502874, 0.186440],
    [0.637160, 0.840715, 1.000000, 0.683745, 0.455052, 0.513472],
    [0.051500, 0.046804, 0.045656, 1.000000, 0.512444, 0.057122],
    [0.054774, 0.039055, 0.029331, 0.494674, 1.000000, 0.067014],
    [0.494776, 0.148405, 0.339219, 0.565150, 0.686843, 1.000000]
])

# Coefficients matrix (for minimax)
coeffs = np.array([[1.47167035,  2.69560101,  2.30973449, 28.57632219, 26.86799411,  2.97442066],
                   [2.69559897,  1.74367916,  2.07404297,
                       37.25506036, 44.6468276,  11.74951997],
                   [2.30973436,  2.07404312,  1.447178,
                       31.69773617, 49.33882014, 4.26621219],
                   [28.57612322, 37.25491759, 31.69743304,
                       21.67316862, 43.81303866, 38.34915626],
                   [26.86804589, 44.64675875, 49.33953847,
                       43.81303367, 22.45172878, 32.68840099],
                   [2.9744174, 11.74946372, 4.26620562, 38.34940921, 32.68829817, 2.1905805]])


# Build pairs including diagonals
n_features = M.shape[0]
pair_indices = list(combinations_with_replacement(
    range(n_features), 2))  # includes (i,i)
n_pairs = len(pair_indices)


# Example: upper-triangular coefficients flattened
coeffs_ij = np.array([coeffs[i, j] for i, j in pair_indices])

# Feature-to-pairs incidence matrix (for goal constraints)
# M_pairs = np.zeros((n_features, n_pairs))
# for i, (p, q) in enumerate(pair_indices):
#     M_pairs[p, i] = 1
#     M_pairs[q, i] = 1

M_pairs = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1.0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0,
                        1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0,
                        1.0, 0, 0, 1.0, 1.0, 1.0, 0, 0, 0],
                    [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0,
                        0, 1.0, 0, 0, 1.0, 0, 1.0, 1.0, 0],
                    [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 1.0, 1.0]])

# for all pairs (i,i) add some epsilon to M
for idx, (i, j) in enumerate(pair_indices):
    if i == j:
        # small epsilon, ensures that singletons will be preferred
        M_pairs[i, idx] += 0.00001


def solve_minimax(goal):
    # Variables: v_ij..., t  -> total length = n_pairs + 1
    c = np.zeros(n_pairs + 1)
    c[-1] = 1  # objective: minimize t

    # 1) Goal constraint: M_pairs @ v >= goal
    A_ub = np.hstack([-M_pairs, np.zeros((n_features, 1))])
    b_ub = -goal

    # 2) coeffs * v_ij <= t
    for i, coeff in enumerate(coeffs_ij):
        row = np.zeros(n_pairs + 1)
        row[i] = coeff
        row[-1] = -1
        A_ub = np.vstack([A_ub, row])
        b_ub = np.append(b_ub, 0.0)

    # 2b) We want an additional constraint for implicit v_i = sum_j v_ij
    # Let coeffs for that v_i is coeffs_ij[i,i] (the diagonal of coeffs)
    coeffs_diag = np.diag(coeffs)
    # Also for this implicit v_i we want coeffs_diag[i] * v_i <= t
    for i in range(n_features):
        row = np.zeros(n_pairs + 1)
        # sum over all pairs that include i
        for idx, (p, q) in enumerate(pair_indices):
            if i == p or i == q:
                row[idx] = coeffs_diag[i]
        row[-1] = -1
        A_ub = np.vstack([A_ub, row])
        b_ub = np.append(b_ub, 0.0)

    # 3) Normalization: sum v_ij = 1
    A_eq = np.zeros((1, n_pairs + 1))
    A_eq[0, :n_pairs] = 1
    b_eq = np.array([1.0])

    # 4) Bounds
    bounds = [(0, None)] * (n_pairs + 1)

    # Solve
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
            # I think venyl_collector is easy to reach so I reduce remaining_needed by 20
            if attr_id == "vinyl_collector":
                values[attr_id] = (remaining_needed - 10) / remaining_spots

    # define the vector v
    goal = np.array([values.get("underground_veteran", 0.0), values.get("international", 0.0),
                     values.get("fashion_forward", 0.0), values.get(
                         "queer_friendly", 0.0),
                     values.get("vinyl_collector", 0.0), values.get("german_speaker", 0.0)])
    # To make it solvable, we let underground_veteran and fashion_forward be 0
    goal = np.array([0.0, goal[1], 0.0, goal[3], goal[4], goal[5]])
    res = solve_minimax(goal)

    if res.success:
        v = res.x[:-1]   # v_ij variables
        t = res.x[-1]    # minimized maximum
        acceptance_rates = coeffs_ij / t * v

        # Reconstruct 6x6 matrix V
        n_features = len(relativeFrequencies)
        V = np.zeros((n_features, n_features))
        A = np.zeros((n_features, n_features))
        for idx, (i, j) in enumerate(pair_indices):
            V[i, j] = v[idx]
            A[i, j] = acceptance_rates[idx]
        print("\nReconstructed V matrix:\n", V)
        print("\nAcceptance rates matrix:\n", A)
        print("Optimal t (minimized max):", t)
        # M_pairs @ v
        print("M_pairs @ v:", M_pairs @ v)
        print("Goal:", goal)
        # Prognose is as follows:
        # take remaing spots and multiply with t and substract t, then add already admitted
        prognose = rejected_count + remaining_spots * (t - 1)
        print("Prognose admitted after all:", prognose)

        # For a person define the property Matrix which is 6x6 with 1 if person has both attributes i and j
        person_matrix = np.zeros((n_features, n_features))
        person_attributes = next_person.get("attributes", {})
        for i, attr_i in enumerate(["underground_veteran", "international", "fashion_forward",
                                   "queer_friendly", "vinyl_collector", "german_speaker"]):
            for j, attr_j in enumerate(["underground_veteran", "international", "fashion_forward",
                                       "queer_friendly", "vinyl_collector", "german_speaker"]):
                if i <= j and person_attributes.get(attr_i, False) and person_attributes.get(attr_j, False):
                    person_matrix[i, j] = 1
        print("Person attribute matrix:\n", person_matrix)
        # Now we collect all urgent attributes (where goal > 0, acceptance rate > 0.99, and v_ij > 0)
        urgent_indices = set()
        normal_indices = set()
        acceptance_sum = 0.0
        for i in range(n_features):
            for j in range(n_features):
                if person_matrix[i, j] == 1:
                    acceptance_sum += A[i, j]
                    if A[i, j] > 0.99:
                        urgent_indices.add((i, j))
                    if A[i, j] > 0.001:
                        normal_indices.add((i, j))
        print("Urgent attribute pairs (indices):", urgent_indices)
        print("Normal attribute pairs (indices):", normal_indices)
        # Accept if any urgent or normal pair is present
        if urgent_indices or normal_indices:
            print("Accepting the person based on urgent/normal attributes.")
            return True
        return False

    else:
        print("No feasible solution found.")
        print("Goal:", goal)
        return True


if __name__ == "__main__":
    run_algorithm(algo_03_decision, scenario=3, algo_name="algo_03")
