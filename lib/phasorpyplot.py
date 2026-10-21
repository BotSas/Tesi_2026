import numpy as np
import pandas as pd
import tifffile as tiff
import sys
import plotly.graph_objects as go
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
import xarray as xr

# stolen from phasor.py
from phasorpy.plot import PhasorPlot  
from phasorpy.datasets import fetch

# personal stufF
import lib.randgen
import lib.my_stats




# PHASOR PLOT AND NOISE ADDITION

#plotting functions

def data_and_phasor_plot(xe, n, G, S, T_laser, exp1data):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))

    # hist + plot
    bin_centers = 0.5 * (xe[:-1] + xe[1:])    # select bin centers

    # use the following line with a lower binning to show a clearer histogram of the data
    # ax1.hist(exp1data, bins = xe, density = False, alpha = 0.3, color = 'steelblue', edgecolor = 'midnightblue', label = 'Data')

    ax1.plot(bin_centers, n, color = 'purple', marker = '', linestyle = '-', linewidth = 1.5)

    ax1.set_xlabel('t (ns)', fontsize = 12)
    ax1.set_ylabel('counts', fontsize = 12)
    # ax1.legend(fontsize = 11, frameon = True, shadow = True)
    ax1.grid(True, which = "both", linestyle = "--", alpha = 0.5)
    ax1.set_xlim(0, xe[-1])



    # phasor plane
    ω_laser = 1000.0 / T_laser  # Calcolo della frequenza in MHz
    plot = PhasorPlot(frequency = ω_laser, ax = ax2)

    plot.plot([G], [S], marker = 'x', markersize = 10, markeredgewidth = 2, 
              color = 'purple', label = 'Fasore (Dati)')

    #ax2.legend(loc = 'upper right', fontsize = 11, frameon = True, shadow = True)

    # ax1.set_yscale('log')         #  looks terrible for some reason

    plt.tight_layout()
    plt.show()

#######################################################################################################



def plot_decay(xe, n):                  # plots exponential data alone

    fig, ax = plt.subplots(figsize=(7, 5))
    t = 0.5 * (xe[:-1] + xe[1:])

    ax.step(t, n, where = 'mid', color = 'purple', linewidth = 1.5)
    ax.set(xlabel = 't (ns)', ylabel = 'counts')
    ax.grid(alpha = 0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()


#####################################################################################################




def universal_phasor_plot(T_laser, points, lines = None, figsize = (8, 8), titolo = None):     # Plots every single phasor plot imaginable

    """    
    - points: dict list for markers [{'G': 0.5, 'S': 0.4, 'label': 'IRF', 'color': 'darkorange', 'marker': 'o'}]
    - lines: dict list for lines [{'G': [G1, G2], 'S': [S1, S2], 'color': 'gray', 'ls': '--'}]
    """

    fig, ax = plt.subplots(figsize = figsize)

    ω_laser = 1000.0 / T_laser
    plot = PhasorPlot(frequency = ω_laser, ax = ax, title = titolo)
    
    #LINES 
    if lines:
        for line in lines:
            line_style = line.copy()
            G_coords = line_style.pop('G')
            S_coords = line_style.pop('S')

            line_style.setdefault('zorder', 1) 
            
            ax.plot(G_coords, S_coords, **line_style)
            
    # MARKERS 
    for pt in points:
        pt_style = pt.copy()
        G_val = pt_style.pop('G')
        S_val = pt_style.pop('S')

        xerr = pt_style.pop('xerr', None)
        yerr = pt_style.pop('yerr', None)

        pt_style.setdefault('marker', 'o')
        pt_style.setdefault('markersize', 8)
        pt_style.setdefault('zorder', 3)

        # Error bars, only if points have xerr and/or yerr specified
        if xerr is not None or yerr is not None:
            error_style = {
                key: pt_style[key]
                for key in ('color', 'alpha', 'zorder')
                if key in pt_style
            }
            error_style.setdefault('ecolor', pt_style.get('color', 'black'))
            error_style.setdefault('capsize', 0)

            ax.errorbar(
                G_val,
                S_val,
                xerr = xerr,
                yerr = yerr,
                fmt = 'none',
                **error_style
            )

        plot.plot([G_val], [S_val], **pt_style)
        
    # Cosmetic stuff
    ax.set_xlabel('G, real', fontsize = 12)
    ax.set_ylabel('S, imaginary', fontsize = 12)
    ax.legend(loc = 'upper right', frameon = True, fontsize = 11, shadow = True)

    plt.tight_layout()
    plt.show()



#####################################################################################################


# WORKING ON AN IMPORTED IMAGE


def image_flattener(image):

    ''' Plots a 2d image with the amount of photons per pixel in total time '''

    flat_image = image.sum(dim = "time")      # sum of total photons
    flat_pixels = flat_image.stack(pixel = ("x", "y"))

    plt.imshow(flat_image.values, cmap = "magma")         # alternatives I like: plasma, viridis, gray
    plt.colorbar(label = "total counts")

    # max_index = flat_pixels.argmax(dim="pixel").item()               # auto select point with highest counts
    # max_pixel = np.unravel_index(max_index, flat_image.shape)

    # plt.scatter(              # commented highest count pixel selection, pointless
    #     max_pixel[1],
    #     max_pixel[0],
    #     facecolors = "none",
    #     edgecolors = "cyan",
    #     marker = "o",
    #     s = 100,
    #     linewidths = 2,
    # )

    plt.show()




def pixel_plot(image, pixel):

    ''' Given a [t, x, y] image, plots the exponential decay in a single pixel '''

    time = np.linspace(0, 12.5, image.sizes["time"])
    decay = image.isel(x = pixel[0], y = pixel[1]).values

    plt.figure(figsize = (17, 5))
    plt.step(time, decay, where = "mid", color = "plum", linewidth = 1.5)

    plt.yscale('log')   # I am using log scale on y because it's what every single paper I found did the same

    # 12.5 ns / 4096 bins = 0.00305 ns per bin
    plt.xlabel("time (ns)")
    plt.ylabel("counts (Log)")
    plt.grid(True, which = "major", linestyle = "-", alpha = 0.5)

    plt.show()




def global_image_plot(image):  

    ''' This plots the exponential decay of the entire image - uses pixel_plot '''

    global_exp = image.sum(dim = ("x", "y"))
    reshape = global_exp.expand_dims({"x": [0], "y": [0]})  
    pixel_plot(reshape, pixel = (0, 0))


###########################################################################################################