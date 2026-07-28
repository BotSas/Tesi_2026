import numpy as np
import random
import scipy as sp
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
import sys
import math

import lib.randgen
import lib.my_stats
import lib.funclib




def generate_gaus_bm():
    x1, x2 = lib.randgen.generate_uniform(2)
    a, b = g1(x1, x2), g2(x1, x2)    # restituisce i due numeri pseudo-casuali calcolati con g1 e g2
    return [a, b]    # li salvo in una lista in modo da poter spacchettare più facilmente i due valori nel blocco dopo con
                     # (x_i + lib_esame.generate_gaus_bm())




# definisco le due funzioni g1 e g2

def g1(x1, x2):
    return np.sqrt(-2 * math.log(x1)) * np.cos(2 * np.pi * x2)


def g2(x1, x2):
    return np.sqrt(-2 * math.log(x1)) * np.sin(2 * np.pi * x2)