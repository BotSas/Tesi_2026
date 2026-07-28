import numpy as np
#from random import random
import scipy as sp
from scipy.stats import chi2
import sys

def stat(sample):

        mu = np.average(sample)
        var = np.var(sample)
        stdDev = np.std(sample) 
        skewness = sp.stats.skew(sample)
        kurtosis = sp.stats.kurtosis(sample)

        return {"mu": mu, "var": var, "stdDev": stdDev, "sk": skewness, "ku": kurtosis}

def sturges(N):
        '''returns the surges function applied to the sample length'''

        try:
                N = len(N)
        except:
                pass
        
        finally:
                return int(np.ceil(1 + 3.322 * np.log(N)))


if __name__ == "__main__":
        print(sturges([i for i in range(1, 1484)]))
        print(sturges(1484))
        

        
def minuit_stats(m):
    print (f"chi-quadro: {m.fval:.2f}")
    print (f"chi-quadro ridotto: {m.fval / m.ndof:.2f}")
    print (f"dof: {m.ndof:.2f}")
    print (f"p-value: {(1. - chi2.cdf (m.fval, df = m.ndof)):.2f}")
    
    
    
#!/usr/bin/python

# from myrand import generate_range, rand_range
# from math import sqrt

# def integral_HOM (func, xMin, xMax, yMax, N_evt) :

#     x_coord = generate_range (xMin, xMax, N_evt)
#     y_coord = generate_range (0., yMax, N_evt)

#     points_under = 0
#     for x, y in zip (x_coord, y_coord):
#         if (func (x) > y) : points_under = points_under + 1 

#     A_rett = (xMax - xMin) * yMax
#     frac = float (points_under) / float (N_evt)
#     integral = A_rett * frac
#     integral_unc = A_rett**2 * frac * (1 - frac) / N_evt
#     return integral, integral_unc


# # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


# def integral_CrudeMC (g, xMin, xMax, N_rand) :
#     somma     = 0.
#     sommaQ    = 0.    
#     for i in range (N_rand) :
#        x = rand_range (xMin, xMax)
#        somma += g(x)
#        sommaQ += g(x) * g(x)     
     
#     media = somma / float (N_rand)
#     varianza = sommaQ /float (N_rand) - media * media 
#     varianza = varianza * (N_rand - 1) / N_rand
#     lunghezza = (xMax - xMin)
#     return media * lunghezza, sqrt (varianza / float (N_rand)) * lunghezza
                                         
