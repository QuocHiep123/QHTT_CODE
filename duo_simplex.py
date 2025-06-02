import numpy as np
from inputt import get_data

MAX_MODE = 'MAX'
MIN_MODE = 'MIN'

class SimplexMethod:
    def __init__(self, c, a, b, mode):
        self.main_variables_count = a.shape[1]
        self.restrictions_count = a.shape[0]
        self.variables_count = self.main_variables_count + self.restrictions_count
        self.mode = mode

        self.c = np.concatenate([c, np.zeros((self.restrictions_count + 1))])
        self.f = np.zeros((self.variables_count + 1))
        self.basis = [
            i + self.main_variables_count for i in range(self.restrictions_count)]
        self.init_table(a, b)

    def init_table(self, a, b):
        self.table = np.zeros((self.restrictions_count, self.variables_count + 1))
        for i in range(self.restrictions_count):
            for j in range(self.main_variables_count):
                self.table[i][j] = a[i][j]
            for j in range(self.restrictions_count):
                self.table[i][j + self.main_variables_count] = int(i == j)
            self.table[i][-1] = b[i]

    def get_negative_b_row(self):
        row = -1
        for i, a_row in enumerate(self.table):
            if a_row[-1] < 0 and (row == -1 or abs(a_row[-1]) > abs(self.table[row][-1])):
                row = i
        return row

    def get_negative_b_column(self, row):
        column = -1
        for i, aij in enumerate(self.table[row][:-1]):
            if aij < 0 and (column == -1 or abs(aij) > abs(self.table[row][column])):
                column = i
        return column

    def remove_negative_b(self):
        while True:
            row = self.get_negative_b_row()
            if row == -1:
                return True
            column = self.get_negative_b_column(row)
            if column == -1:
                return False
            self.gauss(row, column)
            self.calculate_f()
            print('\nLeaving variable has been removed in row:', row + 1)
            self.print_table()

    def gauss(self, row, column):
        self.table[row] /= self.table[row][column]
        for i in range(self.restrictions_count):
            if i != row:
                self.table[i] -= self.table[row] * self.table[i][column]
        self.basis[row] = column

    def calculate_f(self):
        for i in range(self.variables_count + 1):
            self.f[i] = -self.c[i]
            for j in range(self.restrictions_count):
                self.f[i] += self.c[self.basis[j]] * self.table[j][i]

    def get_relations(self, column):
        q = []
        for i in range(self.restrictions_count):
            if self.table[i][column] == 0:
                q.append(np.inf)
            else:
                q_i = self.table[i][-1] / self.table[i][column]
                q.append(q_i if q_i >= 0 else np.inf)
        return q

    def get_solve(self):
        y = np.zeros((self.variables_count))
        for i in range(self.restrictions_count):
            y[self.basis[i]] = self.table[i][-1]
        return y

    def print_final_result(self):
        solution = self.get_solve()
        print("\n===== FINAL RESULT =====")
        for i in range(self.main_variables_count):
            print(f"x{i+1} = {solution[i]:.4f}")
        # Giá trị tối ưu (chuyển lại đúng dấu nếu bài toán gốc là MIN)
        if self.mode == MAX_MODE:
            optimal_value = np.dot(self.c[:self.main_variables_count], solution[:self.main_variables_count])
        else:
            optimal_value = -np.dot(self.c[:self.main_variables_count], solution[:self.main_variables_count])
        print(f"Optimal value: {optimal_value:.4f}")

    def solve(self):
        print('\nIteration 0')
        self.calculate_f()
        self.print_table()
        if not self.remove_negative_b():
            print('Solve does not exist')
            return False
        iteration = 1
        while True:
            self.calculate_f()
            print('\nIteration', iteration)
            self.print_table()
            if all(fi >= 0 if self.mode == MAX_MODE else fi <= 0 for fi in self.f[:-1]):
                break
            column = (np.argmin if self.mode == MAX_MODE else np.argmax)(self.f[:-1])
            q = self.get_relations(column)
            if all(qi == np.inf for qi in q):
                print('Solve does not exist')
                return False
            self.gauss(np.argmin(q), column)
            iteration += 1
        self.print_final_result()   # <-- Gọi in kết quả cuối cùng ở đây
        return True

    def print_table(self):
        print('     |' + ''.join(['   y%-3d |' % (i + 1)
              for i in range(self.variables_count)]) + '    b   |')
        for i in range(self.restrictions_count):
            print('%4s |' % ('y' + str(self.basis[i] + 1)) + ''.join(
                [' %6.2f |' % aij for j, aij in enumerate(self.table[i])]))
        print('   F |' + ''.join([' %6.2f |' % aij for aij in self.f]))
        print('   y |' + ''.join([' %6.2f |' % xi for xi in self.get_solve()]))

    def print_coef(self, ai, i):
        if ai == 1:
            return 'y%d' % (i + 1)
        if ai == -1:
            return '-y%d' % (i + 1)
        return '%.2fy%d' % (ai, i + 1)

    def print_task(self, full=False):
        print(' + '.join(['%.2fy%d' % (ci, i + 1) for i, ci in enumerate(
            self.c[:self.main_variables_count]) if ci != 0]), '-> ', self.mode)
        for row in self.table:
            if full:
                print(' + '.join([self.print_coef(ai, i) for i, ai in enumerate(
                    row[:self.variables_count]) if ai != 0]), '=', row[-1])
            else:
                print(' + '.join([self.print_coef(ai, i) for i, ai in enumerate(
                    row[:self.main_variables_count]) if ai != 0]), '<=', row[-1])

    @classmethod
    def from_data(cls, target_func, min_max_func, coefficients, constraint, sig_constraint, val_constraint):
        """
        Khởi tạo và chuẩn hóa dữ liệu từ input gốc (chuỗi), tự động chuyển MIN thành MAX, chuẩn hóa ràng buộc.
        """
        if min_max_func.upper() == 'MIN':
            c = -np.array([float(x) for x in target_func])
            mode = MAX_MODE
        else:
            c = np.array([float(x) for x in target_func])
            mode = min_max_func.upper()
        a, b = standardize_problem(coefficients, sig_constraint, val_constraint)
        return cls(c, a, b, mode)

def standardize_problem(coefficients, sig_constraint, val_constraint):
    """
    Chuyển tất cả ràng buộc về dạng <=
    """
    A = []
    b = []
    for row, sig, val in zip(coefficients, sig_constraint, val_constraint):
        row = [float(x) for x in row]
        val = float(val)
        if sig.strip() == '>=':
            row = [-x for x in row]
            val = -val
        A.append(row)
        b.append(val)
    return np.array(A), np.array(b)



