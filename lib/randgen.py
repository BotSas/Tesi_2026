#!/usr/bin/python

import random
from math import sqrt, exp, log
import numpy as np


def generate_uniform (N, seed = 0.) :
    '''
    generazione di un array di N numeri pseudo-casuali distribuiti fra 0 ed 1
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (random.random ())
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_range (xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    '''
    return xMin + random.random () * (xMax - xMin)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_range (xMin, xMax, N, seed = 0.) :
    '''
    generazione di un array di N numeri pseudo-casuali distribuiti fra xMin ed xMax
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_range (xMin, xMax))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TAC (f, xMin, xMax, yMax) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo try and catch
    '''
    x = rand_range (xMin, xMax)
    y = rand_range (0, yMax)
    while (y > f (x)) :
        x = rand_range (xMin, xMax)
        y = rand_range (0, yMax)
    return x


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TAC (f, xMin, xMax, yMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo try and catch, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TAC (f, xMin, xMax, yMax))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TCL (xMin, xMax, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    su un intervallo fissato
    '''
    y = 0.
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum ;
    return y ;


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TCL (xMin, xMax, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TCL_ms (mean, sigma, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    note media e sigma della gaussiana
    '''
    y = 0.
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum ;
    return y ;


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TCL_ms (mean, sigma, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, note media e sigma della gaussiana,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist




# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


# funzione che calcola l'integrale di una funzione con hit or miss (area sottesa)

def integral_hit_or_miss(function, xMin, xMax, yMin, yMax, N_of_events):
    
    x_coord = generate_range(xMin, xMax, N_of_events, seed = 0.)
    y_coord = generate_range(yMin, yMax, N_of_events, seed = 0.)
    
    hits = 0
    for x, y in zip (x_coord, y_coord):
        if (function(x) > y) : hits += 1  
    
    A_rettangolo = (xMax - xMin) * yMax
    frac = hits / N_of_events
    
    integral = A_rettangolo * frac
    integral_unc = np.power(A_rettangolo, 2) * frac * (1 - frac) / N_of_events
    return integral, integral_unc

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

def exp_pdf (x, tau) :
    '''
    the exponential probability density function
    '''
    if tau == 0. : return 1.
    return exp (-1 * x / tau) / tau


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def likelihood (theta, pdf, sample) :
    '''
    the likelihood function calculated
    for a sample of independent variables idendically distributed 
    according to their pdf with parameter theta
    '''
    risultato = 1.
    for x in sample:
        risultato = risultato * pdf (x, theta)
    return risultato


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def loglikelihood (theta, pdf, sample) :
    '''
    the log-likelihood function calculated
    for a sample of independent variables idendically distributed 
    according to their pdf with parameter theta
    '''
    risultato = 0.
    for x in sample:
        if (pdf (x, theta) > 0.) : risultato = risultato + log(pdf (x, theta))    
    return risultato