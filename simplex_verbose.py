import numpy as np
from fractions import Fraction

def print_tableau(tableau, basis, var_names):
    m, n = tableau.shape
    print("\nCurrent Tableau:")
    header = ["Basic"] + var_names + ["RHS"]
    print("{:<10}".format(header[0]), end="")
    for name in header[1:]:
        print("{:>12}".format(name), end="")
    print()
    for i, row in enumerate(tableau[:-1]):
        print("{:<10}".format(var_names[basis[i]]), end="")
        for val in row:
            print("{:>12.3f}".format(float(val)), end="")
        print()
    print("   z      ", end="")
    for val in tableau[-1]:
        print("{:>12.3f}".format(float(val)), end="")
    print("\n")

def simplex_verbose(target_fuct, min_max_fuct, coefficients, constraint, sig_constraint, val_constraint):
    '''
    Solve a linear programming problem using the simplex algorithm.
    Supports <=, >=, = constraints and free variables.
    '''
    try:
        # Convert inputs to floats
        c = np.array([float(x) for x in target_fuct])
        b = np.array([float(val) for val in val_constraint])
        A = np.array([[float(x) for x in row] for row in coefficients])

        # Handle free variables
        free_vars = [i for i, cons in enumerate(constraint) if cons.strip().lower() == 'free']
        new_c = []
        var_names = []
        idx_map = {}
        cnt = 0
        for j in range(len(c)):
            if j in free_vars:
                new_c.extend([c[j], -c[j]])
                var_names.extend([f"x{j+1}_plus", f"x{j+1}_minus"])
                idx_map[j] = (cnt, cnt+1)
                cnt += 2
            else:
                new_c.append(c[j])
                var_names.append(f"x{j+1}")
                idx_map[j] = (cnt,)
                cnt += 1

        # Process constraints
        new_A = []
        new_sig = []
        for i, (row, sig) in enumerate(zip(A, sig_constraint)):
            new_row = []
            for j in range(len(row)):
                if j in free_vars:
                    new_row.extend([float(row[j]), -float(row[j])])
                else:
                    new_row.append(float(row[j]))
            if sig.strip() in ['>=', '>']:
                new_A.append([-x for x in new_row])
                b[i] = -b[i]
                new_sig.append('<=')
            else:
                new_A.append(new_row)
                new_sig.append(sig.strip())
        A = np.array(new_A)
        c = np.array(new_c)

        # Print transformed constraints
        print("Transformed constraints:")
        for i in range(len(new_sig)):
            print(f"{' + '.join(f'{A[i,j]:.1f}{var_names[j]}' for j in range(len(var_names)))} {new_sig[i]} {b[i]:.1f}")

        # Add slack variables
        m, n = A.shape
        slack = np.eye(m)
        A = np.hstack([A, slack])
        c = np.hstack([c, np.zeros(m)])
        var_names = var_names + [f"s{i+1}" for i in range(m)]

        # Initialize tableau
        tableau = np.hstack([A, b.reshape(-1, 1)])
        c_row = np.hstack([-c, np.array([0.0])])
        tableau = np.vstack([tableau, c_row])
        basis = list(range(n, n + m))

        # Handle minimization
        if min_max_fuct.upper() == "MIN":
            tableau[-1, :-1] = -tableau[-1, :-1]

        # Simplex iterations
        step = 1
        while True:
            print(f"--- Step {step} ---")
            print_tableau(tableau, basis, var_names)
            last_row = tableau[-1, :-1]
            if all(last_row >= -1e-10):
                print("Optimal solution found.")
                break

            entering = np.argmin(last_row)
            print(f"Entering variable: {var_names[entering]} (column {entering})")

            ratios = []
            for i in range(m):
                if tableau[i, entering] > 1e-10:
                    ratios.append(tableau[i, -1] / tableau[i, entering])
                else:
                    ratios.append(np.inf)
            leaving = np.argmin(ratios)
            if ratios[leaving] == np.inf:
                print("Unbounded solution.")
                return None, None

            print(f"Leaving variable: {var_names[basis[leaving]]} (row {leaving})")
            pivot = tableau[leaving, entering]
            print(f"Pivot element: {float(pivot):.3f}\n")

            # Pivot with exact arithmetic
            tableau[leaving, :] /= pivot
            for i in range(m + 1):
                if i != leaving:
                    tableau[i, :] -= tableau[i, entering] * tableau[leaving, :]
            basis[leaving] = entering
            step += 1

        # Extract solution
        all_var_values = np.zeros(len(var_names))
        for i in range(m):
            if basis[i] < len(var_names):
                all_var_values[basis[i]] = tableau[i, -1]

        print("\nOptimal solution (all variables):")
        for name, value in zip(var_names, all_var_values):
            print(f"{name} = {Fraction(value).limit_denominator(100)}")

        # Original variables
        print("\nOptimal solution for original variables:")
        for j in range(len(constraint)):
            if j in free_vars:
                idx_plus, idx_minus = idx_map[j]
                x_val = all_var_values[idx_plus] - all_var_values[idx_minus]
                print(f"x{j+1} = {Fraction(x_val).limit_denominator(100)} (={var_names[idx_plus]} - {var_names[idx_minus]})")
            else:
                idx = idx_map[j][0]
                print(f"x{j+1} = {Fraction(all_var_values[idx]).limit_denominator(100)}")

        # Compute final value
        final_value = tableau[-1, -1]
        if min_max_fuct.upper() == "MIN":
            final_value = final_value
        else:
            final_value = -final_value
        print(f"Optimal value: {Fraction(final_value).limit_denominator(100)} ({float(final_value):.3f})")
        return all_var_values, final_value

    except Exception as e:
        print(f"Error solving the problem: {e}")
        return None, None