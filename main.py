import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pulp import *
import math
import random
from typing import List
from simplex_verbose import simplex_verbose
from bland_algorithm import bland_algorithm
from two_phase_simplex import two_phase_simplex_verbose
from duo_simplex import  MAX_MODE, MIN_MODE, standardize_problem, SimplexMethod
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inputt import get_data


if __name__ == "__main__":
    target_func, min_max_func, coefficients, constraint, sig_constraint, val_constraint = get_data()
    # simplex = SimplexMethod.from_data(target_func, min_max_func, coefficients, constraint, sig_constraint, val_constraint)
    # simplex.print_task()
    # simplex.solve()
    # res1 = two_phase_simplex_verbose(target_func, min_max_func, coefficients, constraint, sig_constraint, val_constraint)
    # print("\nResult from Two-Phase Simplex:", res1)
    # res2 = bland_algorithm(target_func, min_max_func, coefficients, constraint, sig_constraint, val_constraint)
    # print("\nResult from Bland's Algorithm:", res2)
    res3 = simplex_verbose(target_func, min_max_func, coefficients, constraint, sig_constraint, val_constraint)
    print("\nResult from Simplex Verbose:", res3)
    
