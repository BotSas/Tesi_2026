import numpy as np
import pandas as pd
import tifffile as tiff
import sys
import plotly.express as px
import plotly.graph_objects as go
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
import xarray as xr

# stolen from phasor.py
from phasorpy.plot import PhasorPlot  
from phasorpy.datasets import fetch
from phasorpy.filter import phasor_filter_median, phasor_threshold


# personal stufF
import lib.randgen
import lib.my_stats

# program specific
import lib.phasorpyplot as phplt
import lib.phasorfunctions as phf

# graphics
import io
import panel as pn
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
from itertools import cycle
import colorcet



# This is a set of (mostly graphical) functions that needed to be rewritten to work with the panel app.
# Most of them are based on other functions in lib/phasorfunctions.py



def generate_decay(time_bins, tau):

    '''Generates a single exp decay, normalized to the area'''

    decay = np.exp(-time_bins / tau)
    return decay / np.sum(decay)



def pure_phasor(tau, T_laser, harm=1):

    '''C
    alculates phasor coords for pure tau the same as phasor_calc but takes tau as an input 
    and does not calculate fft. This way you do not add unnecessary errors.
     '''

    omega = 2 * np.pi / T_laser * harm
    omega_tau = omega * tau
    return 1 / (1 + omega_tau**2), omega_tau / (1 + omega_tau**2)



# Universal Phasor Plot (Bokeh/HoloViews) 

def create_phasor_plot(T_laser, points, lines=None):

    '''
    Same as universal_phasor_plot (see lib/phasorpyplot.py), just works with hv/bokeh.
    Takes dicts of lines and points as inputs
    '''

    freq_mhz = 1000.0 / T_laser
    title = f"Phasor plot ({freq_mhz:.0f} MHz)"
    
    # Draws semicircle
    g_curve = np.linspace(0, 1, 500)
    s_curve = np.sqrt(0.25 - (g_curve - 0.5)**2)
    plot = hv.Curve((g_curve, s_curve)).opts(color="gray", line_width=1.2)
    
    # lifetimes tags on semicircle
    omega = 2 * np.pi / T_laser
    lifetimes = [0, 0.5, 1, 2, 4, 8]
    for tau in lifetimes:
        g_t = 1 / (1 + (omega * tau)**2)
        s_t = (omega * tau) / (1 + (omega * tau)**2)
        
        label = f"{tau} ns" if tau == 1 else f"{tau}"
        
        # Offset so text is outside semicircle
        offset_g = (g_t - 0.5) * 0.05
        offset_s = s_t * 0.05 + 0.015
        if tau == 0:
            offset_g, offset_s = 0.02, 0.0
            
        plot *= hv.Text(g_t + offset_g, s_t + offset_s, label).opts(
            text_font_size = "9pt", text_color = "gray"
        )
        
    # LINES
    if lines:
        for line in lines:
            line_style = line.copy()
            G_coords = line_style.pop('G')
            S_coords = line_style.pop('S')
            
            ls = line_style.get('ls', '-')
            line_dash = 'dashed' if ls == '--' else 'dotted' if ls == ':' else 'solid'
            
            plot *= hv.Curve((G_coords, S_coords)).opts(
                color = line_style.get('color', 'gray'), line_dash = line_dash, line_width = 1.5
            )
            
    # POINTS
    for pt in points:
        pt_style = pt.copy()
        G_val = pt_style.pop('G')
        S_val = pt_style.pop('S')
        
        marker = pt_style.get('marker', 'o')
        hv_marker = 'circle'
        if marker == '*': hv_marker = 'star'
        elif marker == 'x': hv_marker = 'x'
        
        size = pt_style.get('markersize', 8)
        if hv_marker == 'star': size += 6
        elif hv_marker == 'x': size += 2
        
        color = pt_style.get('color', 'black')
        fill_color = color
        line_width = pt_style.get('linewidth', 1)
        
        # Empty circles (only used for 'target')
        if pt_style.get('facecolor') == 'none' or pt_style.get('mfc') == 'none':
            fill_color = 'white'

            if 'linewidth' not in pt_style: line_width = 2
                
        plot *= hv.Scatter([(G_val, S_val)], label=pt_style.get('label', '')).opts(
            color=color, fill_color=fill_color, marker=hv_marker, 
            size=size, line_color=color, line_width=line_width
        )
        
    return plot.opts(
            title = title, xlim = (-0.05, 1.05), ylim = (-0.05, 0.75),
            xlabel = "G, real", ylabel = "S, imaginary", 
            width = 650, height = 550, show_grid = False, data_aspect = 1,
            legend_position = 'bottom', show_frame = True,
            fontsize = {'labels': 12, 'legend': 11, 'title': 14}
        )
 