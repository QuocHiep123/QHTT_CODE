import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pulp import *
import math
import random
import time
from typing import List, Tuple, Union

def get_data() -> Tuple[List[str], str, List[List[str]], List[str], List[str], List[str]]:
    """
    Returns the input data for the linear programming problem.
    
    Returns:
        Tuple containing:
        - target_funct: List of objective function coefficients
        - min_max_funct: Optimization goal ("MIN" or "MAX")
        - coefficients: Matrix of constraint coefficients
        - constraint: List of variable constraints
        - sig_constraint: List of constraint signs
        - val_constraint: List of constraint values
    """
    # Objective function coefficients
    target_funct = ['-2', '-1']
    
    # Optimization goal (MIN or MAX)
    min_max_funct = "MIN"
    
    # Variable constraints (non-negativity)
    constraint = ['>=0', '>=0']
    
    # Coefficients matrix for constraints
    coefficients = [
        ['1', '2'],    # -2x1 + x2 <= 1
        ['1', '-1'],    # x1 - x2 >= -2
        
    ]
    
    # Constraint signs (<=, >=, or =)
    sig_constraint = ['<=', '>=']
    
    # Right-hand side values of constraints
    val_constraint = ['6', '3']

    return target_funct, min_max_funct, coefficients, constraint, sig_constraint, val_constraint



