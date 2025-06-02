import numpy as np
from fractions import Fraction

def print_tableau(tableau, basis, var_names):
    m, n = tableau.shape
    print("\nTableau:")
    print("Basic".ljust(8), end="")
    for name in var_names:
        print(f"{name:>8}", end="")
    print(f"{'RHS':>8}")
    for i, row in enumerate(tableau[:-1]):
        print(f"{var_names[basis[i]]:8}", end="")
        for val in row:
            print(f"{val:8.3f}", end="")
        print()
    print("   z    ", end="")
    for val in tableau[-1]:
        print(f"{val:8.3f}", end="")
    print("\n")

def pivot(tableau, row, col):
    tableau[row] = tableau[row] / tableau[row, col]
    for i in range(tableau.shape[0]):
        if i != row:
            tableau[i] = tableau[i] - tableau[i, col] * tableau[row]
    return tableau

def phase_simplex(tableau, basis, var_names, phase_name="Simplex"):
    m, n = tableau.shape
    step = 1
    while True:
        print(f"--- {phase_name}, Step {step} ---")
        print_tableau(tableau, basis, var_names)
        last_row = tableau[-1, :-1]
        if all(last_row >= -1e-10):
            print(f"{phase_name} optimal.")
            break
        entering = np.argmin(last_row)
        print(f"Entering: {var_names[entering]}")
        ratios = []
        for i in range(m-1):
            if tableau[i, entering] > 1e-10:
                ratios.append(tableau[i, -1] / tableau[i, entering])
            else:
                ratios.append(np.inf)
        leaving = np.argmin(ratios)
        if ratios[leaving] == np.inf:
            print(f"Unbounded in {phase_name.lower()}.")
            return None, None, None
        print(f"Leaving: {var_names[basis[leaving]]}")
        pivot_element = tableau[leaving, entering]  # Renamed from pivot to pivot_element
        print(f"Pivot element: {float(pivot_element):.3f}\n")
        tableau = pivot(tableau, leaving, entering)  # Call pivot function
        basis[leaving] = entering
        step += 1
    return tableau, basis, var_names

def two_phase_simplex_verbose(target_funct, min_max_funct, coefficients, constraint, sig_constraint, val_constraint):
    '''
    Three-phase simplex: chuẩn hóa, pha 1 (artificial), pha 2 (gốc).
    '''
    try:
        # --- Phase 0: Chuẩn hóa ---
        c = np.array([float(x) for x in target_funct])
        b = np.array([float(val) for val in val_constraint])
        A = np.array([[float(x) for x in row] for row in coefficients])

        # Xử lý biến tự do (nếu có)
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

        # Chuẩn hóa ràng buộc
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

        print("Transformed constraints:")
        for i in range(len(new_sig)):
            print(f"{' + '.join(f'{A[i,j]:.1f}{var_names[j]}' for j in range(len(var_names)))} {new_sig[i]} {b[i]:.1f}")

        # --- Phase 1: Thêm biến artificial, đặt z = x0 ---
        m, n = A.shape
        slack = np.eye(m)
        artificial = []
        artificial_indices = []
        basis = []
        for i in range(m):
            if b[i] >= 0 and new_sig[i] == '<=':
                basis.append(n + i)  # Slack variable
            else:
                col = np.zeros(m)
                col[i] = 1
                artificial.append(col)
                artificial_indices.append(n + slack.shape[1] + len(artificial) - 1)
                basis.append(n + slack.shape[1] + len(artificial) - 1)
        slack = np.array(slack) if slack.size else np.zeros((m, 0))
        artificial = np.array(artificial).T if artificial else np.zeros((m, 0))
        A1 = np.hstack([A, slack, artificial])
       
        c1 = np.zeros(A1.shape[1])
        for idx in artificial_indices:
            c1[idx] = 1  
        var_names1 = var_names + [f"s{i+1}" for i in range(slack.shape[1])] + [f"a{i+1}" for i in range(artificial.shape[1])]

        tableau = np.hstack([A1, b.reshape(-1, 1)])
        c_row = np.zeros(tableau.shape[1])
        for idx in artificial_indices:
            c_row[idx] = 1
        tableau = np.vstack([tableau, c_row])
      

        print("\n=== PHASE 1: Solve for basic feasible solution (z = x0) ===")
        tableau, basis, var_names1 = phase_simplex(tableau, basis, var_names1, phase_name="Phase 1")
        if tableau is None:
            print("No feasible solution (unbounded in phase 1).")
            return None, None
        if abs(tableau[-1, -1]) > 1e-8:
            print("No feasible solution (artificial variables non-zero).")
            return None, None

        # --- Phase 2: Loại artificial, đặt lại hàm mục tiêu gốc ---
        artificial_indices_set = set(artificial_indices)
        keep_cols = [i for i in range(tableau.shape[1] - 1) if i not in artificial_indices_set]
        A2 = tableau[:-1, keep_cols]
        b2 = tableau[:-1, -1]
        var_names2 = [var_names1[i] for i in keep_cols]

  
        basis2 = []
        for i, idx in enumerate(basis):
            if idx in keep_cols:
                basis2.append(keep_cols.index(idx))
            else:
               
                found = False
                for j in range(len(keep_cols)):
                    if abs(A2[i, j]) > 1e-8:
                        pivot_elem = A2[i, j]
                        A2[i, :] /= pivot_elem
                        b2[i] /= pivot_elem
                        for k in range(A2.shape[0]):
                            if k != i:
                                factor = A2[k, j]
                                A2[k, :] -= factor * A2[i, :]
                                b2[k] -= factor * b2[i]
                        basis2.append(j)
                        found = True
                        break
                if not found:
                    basis2.append(0)  # fallback

       
        c2 = np.zeros(len(var_names2))
        for i, name in enumerate(var_names2):
            if name.startswith("x"):
                idx = int(name.split("_")[0][1:]) - 1
                c2[i] = float(target_funct[idx]) 
                
        
        tableau2 = np.hstack([A2, b2.reshape(-1, 1)])
        c_row2 = np.hstack([c2, np.array([0.0])])
        tableau2 = np.vstack([tableau2, c_row2])

     
        for i in range(len(basis2)):
            tableau2[-1] -= tableau2[-1, basis2[i]] * tableau2[i]

        print("\n=== PHASE 2: Solve original problem ===")
        tableau2, basis2, var_names2 = phase_simplex(tableau2, basis2, var_names2, phase_name="Phase 2")
        if tableau2 is None:
            print("No optimal solution (unbounded in phase 2).")
            return None, None

        # Extract solution
        m2 = len(basis2)
        all_var_values = np.zeros(len(var_names2))
        for i in range(m2):
            if basis2[i] < len(var_names2):
                all_var_values[basis2[i]] = tableau2[i, -1]

        print("\nOptimal solution (all variables):")
        for name, value in zip(var_names2, all_var_values):
            print(f"{name} = {Fraction(value).limit_denominator(100)}")

        print("\nOptimal solution for original variables:")
        for j in range(len(constraint)):
            if j in free_vars:
                idx_plus = var_names2.index(f"x{j+1}_plus") if f"x{j+1}_plus" in var_names2 else None
                idx_minus = var_names2.index(f"x{j+1}_minus") if f"x{j+1}_minus" in var_names2 else None
                x_val = 0
                if idx_plus is not None:
                    x_val += all_var_values[idx_plus]
                if idx_minus is not None:
                    x_val -= all_var_values[idx_minus]
                print(f"x{j+1} = {Fraction(x_val).limit_denominator(100)} (={f'x{j+1}_plus'} - {f'x{j+1}_minus'})")
            else:
                idx = var_names2.index(f"x{j+1}") if f"x{j+1}" in var_names2 else None
                print(f"x{j+1} = {Fraction(all_var_values[idx]).limit_denominator(100) if idx is not None else 0}")

        final_value = tableau2[-1, -1]
        if min_max_funct.upper() == "MAX":
            final_value = -final_value 
        print(f"Optimal value: {Fraction(final_value).limit_denominator(100)} ({float(final_value):.3f})")
        print("Original coefficients:", target_funct)
        print("Phase 2 objective coefficients:", c2)
        print("Final tableau:")
        print(tableau2)
        return all_var_values, final_value

    except Exception as e:
        print(f"Error solving the problem (two-phase simplex): {e}")
        return None, None
