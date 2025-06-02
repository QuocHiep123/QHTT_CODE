import numpy as np
from fractions import Fraction

def print_tableau(tableau, basis, var_names):
    m, n = tableau.shape
    print("\nCurrent Tableau:")
    header = ["Basic"] + var_names + ["RHS"]
    print("{:<6}".format(header[0]), end="")
    for name in header[1:]:
        print("{:>8}".format(name), end="")
    print()
    for i, row in enumerate(tableau[:-1]):
        print("{:<6}".format(var_names[basis[i]]), end="")
        for val in row:
            print("{:>8.3f}".format(val), end="")
        print()
    print("   z  ", end="")
    for val in tableau[-1]:
        print("{:>8.3f}".format(val), end="")
    print("\n")

def bland_algorithm(target_fuct, min_max_fuct, coefficients, constraint, sig_constraint, val_constraint):
    """
    Giải bài toán quy hoạch tuyến tính sử dụng thuật toán Bland, hỗ trợ biến tự do (free variable)
    """
    try:
        # Xử lý free variables
        c = np.array([float(x) for x in target_fuct])
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

        # Xử lý lại ma trận hệ số
        A = []
        b = []
        new_sig = []
        for row, sig, val in zip(coefficients, sig_constraint, val_constraint):
            row = [float(x) for x in row]
            new_row = []
            for j in range(len(row)):
                if j in free_vars:
                    new_row.extend([row[j], -row[j]])
                else:
                    new_row.append(row[j])
            if sig.strip() in ['>=', '>']:
                new_row = [-x for x in new_row]
                b.append(-float(val))
                new_sig.append('<=')
            elif sig.strip() in ['<=', '<']:
                b.append(float(val))
                new_sig.append('<=')
            elif sig.strip() == '=':
                b.append(float(val))
                new_sig.append('=')
            A.append(new_row)
        A = np.array(A)
        b = np.array(b)
        c = np.array(new_c)

        m, n = A.shape
        # Thêm biến phụ cho mọi ràng buộc <=
        slack = np.eye(m)
        A = np.hstack([A, slack])
        c = np.hstack([c, np.zeros(m)])
        var_names = var_names + [f"s{i+1}" for i in range(m)]

        # Nếu là MIN thì đổi dấu hàm mục tiêu
        if min_max_fuct.upper() == "MIN":
            c = -c

        # Khởi tạo ma trận mở rộng (tableau)
        tableau = np.zeros((m + 1, n + m + 1))
        tableau[:-1, :n + m] = A  # Ma trận A đã có biến phụ
        tableau[:-1, -1] = b      # Vector vế phải
        tableau[-1, :n + m] = -c  # Hàng hàm mục tiêu

        # Danh sách biến cơ sở
        basis = list(range(n, n + m))
        step = 1

        print("\nBắt đầu thuật toán Bland:")
        print("------------------------")

        while True:
            print(f"\nBước {step}:")
            print("-" * 50)
            print_tableau(tableau, basis, var_names)

            # Kiểm tra tính tối ưu
            if all(tableau[-1, :-1] >= 0):
                print("Đã tìm thấy nghiệm tối ưu!")
                break

            # Chọn biến vào theo quy tắc Bland
            entering = None
            for j in range(n + m):
                if tableau[-1, j] < 0:
                    entering = j
                    break

            if entering is None:
                break

            print(f"Biến vào: {var_names[entering]}")
            print(f"Cột pivot: {entering}")
            print(f"Giá trị trong hàng z: {tableau[-1, entering]:.3f}")

            # Chọn biến ra theo quy tắc Bland
            leaving = None
            min_ratio = float('inf')
            ratios = []
            for i in range(m):
                if tableau[i, entering] > 0:
                    ratio = tableau[i, -1] / tableau[i, entering]
                    ratios.append(ratio)
                    if ratio < min_ratio:
                        min_ratio = ratio
                        leaving = i
                else:
                    ratios.append(float('inf'))

            if leaving is None:
                print("Bài toán không bị chặn!")
                return None, None

            print(f"\nBiến ra: {var_names[basis[leaving]]}")
            print(f"Hàng pivot: {leaving}")
            print(f"Phần tử pivot: {tableau[leaving, entering]:.3f}")
            print("\nTỷ số kiểm tra:")
            for i in range(m):
                if ratios[i] != float('inf'):
                    print(f"Tỷ số {i+1}: {ratios[i]:.3f}")

            # Cập nhật cơ sở
            basis[leaving] = entering

            # Thực hiện phép biến đổi hàng
            pivot = tableau[leaving, entering]
            tableau[leaving] /= pivot

            for i in range(m + 1):
                if i != leaving:
                    tableau[i] -= tableau[i, entering] * tableau[leaving]

            print("\nMa trận sau khi xoay:")
            print_tableau(tableau, basis, var_names)
            step += 1

        # Tính nghiệm tối ưu cho tất cả các biến
        all_var_values = np.zeros(n + m)
        for i in range(m):
            all_var_values[basis[i]] = tableau[i, -1]

        z = tableau[-1, -1]
        if min_max_fuct.upper() == "MIN":
            z = z  # Đổi dấu lại nếu là bài toán min
        else:
            z = -z

        print("\nNghiệm tối ưu (tất cả các biến):")
        for i, (name, value) in enumerate(zip(var_names, all_var_values)):
            print(f"{name} = {Fraction(value).limit_denominator(100)}")

        # In nghiệm các biến gốc (kể cả biến free)
        print("\nOptimal solution for original variables:")
        for j in range(len(constraint)):
            if j in free_vars:
                idx_plus, idx_minus = idx_map[j]
                x_val = all_var_values[idx_plus] - all_var_values[idx_minus]
                print(f"x{j+1} = {Fraction(x_val).limit_denominator(100)} (={var_names[idx_plus]} - {var_names[idx_minus]})")
            else:
                idx = idx_map[j][0]
                print(f"x{j+1} = {Fraction(all_var_values[idx]).limit_denominator(100)}")

        print(f"Giá trị hàm mục tiêu z = {Fraction(z).limit_denominator(100)} ({float(z):.3f})")

        return all_var_values, z

    except Exception as e:
        print(f"Không thể giải bài toán bằng thuật toán Bland: {e}")
        return None, None

