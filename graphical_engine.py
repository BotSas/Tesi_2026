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
import lib.phasorgraphs as phg
from new_tabs import tab4_view_ext, tab4_card_ext

# graphics
import io
import panel as pn
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
from itertools import cycle
import colorcet



pn.extension("mathjax", sizing_mode="stretch_width")
hv.extension("bokeh")

# This is the main file. It also contains code for tab 1 2 and 3 (single, double, triple exp). For image analysis see other file

# ─── Global widgets ────────────────────────────────────────────────────────────

t_laser_input = pn.widgets.FloatInput(name = "T Laser (ns)", value = 12.5, step = 0.1, width = 160)
bins_input = pn.widgets.IntInput(name = "Time bins", value = 256, step = 16, width = 160)
counts_input = pn.widgets.IntInput(name = "Counts", value=10000, step=1000, width = 160)
harmonic_input = pn.widgets.IntInput(name = "Harmonic N", value = 1, step = 1, start = 1, width = 160)


# ─── Tab 1: Single exponential ───────────────────────────────────────────────

t1_tau = pn.widgets.FloatSlider(name="τ (ns)", start = 0.1, end=10.0, value = 2.5, step = 0.1)
t1_irf_mu = pn.widgets.FloatSlider(name = "IRF μ (ns)", start = 0.0, end = 1, value= 0, step = 0.1)
t1_irf_sig = pn.widgets.FloatSlider(name = "IRF σ (ns)", start = 0.01, end = 0.5, value = 0.1, step = 0.01)
t1_add_noise = pn.widgets.Toggle(name = "Add gaussian noise", button_type = "default", value = False)
t1_snr = pn.widgets.FloatSlider(name = "SNR (dB)", start = 0, end = 40, value = 20, step = 1, disabled = True)
t1_add_noise.param.watch(lambda e: setattr(t1_snr, 'disabled', not e.new), "value")


# ─── Tab 2: Double exponential ────────────────────────────────────────────────

t2_tau1 = pn.widgets.FloatSlider(name="τ1 (ns)", start=0.1, end = 10.0, value = 1.0, step = 0.1)
t2_tau2 = pn.widgets.FloatSlider(name="τ2 (ns)", start=0.1, end = 10.0, value = 4.0, step = 0.1)
t2_f1 = pn.widgets.FloatSlider(name = "τ1 intensity fraction", start = 0.0, end = 1.0, value = 0.5, step = 0.01)
t2_f2 = pn.widgets.FloatSlider(name = "τ2 intensity fraction (Auto)", start=0.0, end=1.0, value=0.5, step = 0.01, disabled = True)
t2_irf_mu = pn.widgets.FloatSlider(name = "IRF μ (ns)", start = 0.0, end = 1, value= 0, step = 0.1)
t2_irf_sig = pn.widgets.FloatSlider(name = "IRF σ (ns)", start = 0.01, end = 0.5, value = 0.1, step = 0.01)
t2_add_noise = pn.widgets.Toggle(name = "Add gaussian noise", button_type = "default", value = False)
t2_snr = pn.widgets.FloatSlider(name = "SNR (dB)", start = 0, end = 40, value = 20, step = 1, disabled = True)
t2_f1.param.watch(lambda e: setattr(t2_f2, 'value', 1.0 - t2_f1.value), "value")
t2_add_noise.param.watch(lambda e: setattr(t2_snr, 'disabled', not e.new), "value")


# ─── Tab 3: Triple exponential ────────────────────────────────────────────────

t3_tau1 = pn.widgets.FloatSlider(name="τ1 (ns)", start = 0.1, end=10.0, value=1.0, step=0.1)
t3_tau2 = pn.widgets.FloatSlider(name="τ2 (ns)", start=0.1, end=10.0, value=4.0, step=0.1)
t3_tau3 = pn.widgets.FloatSlider(name="τ3 (ns)", start=0.1, end=10.0, value=7.0, step=0.1)
t3_f1 = pn.widgets.FloatSlider(name="τ1 intensity fraction", start=0.0, end=1.0, value=0.33, step=0.01)
t3_f2 = pn.widgets.FloatSlider(name="τ2 intensity fraction", start=0.0, end=1.0, value=0.33, step=0.01)
t3_f3 = pn.widgets.FloatSlider(name="τ3 intensity fraction(Auto)", start=0.0, end=1.0, value=0.34, step=0.01, disabled=True)
t3_irf_mu = pn.widgets.FloatSlider(name = "IRF μ (ns)", start = 0.0, end = 1, value= 0, step = 0.1)
t3_irf_sig = pn.widgets.FloatSlider(name = "IRF σ (ns)", start = 0.01, end = 0.5, value = 0.1, step = 0.01)
t3_add_noise = pn.widgets.Toggle(name="Add gaussian noise", button_type="default", value = False)
t3_snr = pn.widgets.FloatSlider(name = "SNR (dB)", start = 0, end = 40, value = 20, step = 1, disabled = True)

_updating_t3 = False
def _clamp_t3_fractions(changed, event):
    global _updating_t3
    if _updating_t3: return
    _updating_t3 = True
    try:
        if changed == "f1" and t3_f1.value + t3_f2.value > 1.0: t3_f2.value = 1.0 - t3_f1.value
        elif changed == "f2" and t3_f1.value + t3_f2.value > 1.0: t3_f1.value = 1.0 - t3_f2.value
        t3_f3.value = max(0.0, 1.0 - t3_f1.value - t3_f2.value)
    finally:
        _updating_t3 = False

t3_f1.param.watch(lambda e: _clamp_t3_fractions("f1", e), "value")
t3_f2.param.watch(lambda e: _clamp_t3_fractions("f2", e), "value")
t3_add_noise.param.watch(lambda e: setattr(t3_snr, 'disabled', not e.new), "value")




# ── GRAPHS ────────────────────────────────────────────
# Functions from phf and phg are defined in phasorgraphs.py and phasorfunctions.py with better explanation on how they work

@pn.depends(
    t_laser_input, bins_input, counts_input, harmonic_input, 
    t1_tau, t1_irf_mu, t1_irf_sig, t1_add_noise, t1_snr
)


def tab1_view(T_laser, n_bins, counts, harm, tau, mu, sig, add_noise, snr):

    time_bins, irf = phf.irf_generation(sig, mu, counts, T_laser, n_bins)
    pure_decay = phg.generate_decay(time_bins, tau)
    convol, noisy_signal = phf.simulation_with_IRF(pure_decay, irf, counts, add_noise, snr)
    
    G_meas, S_meas = phf.phasor_calc(noisy_signal, harm)
    G_irf, S_irf = phf.phasor_calc(irf, harm)
    G_dec, S_dec = phf.phasor_deconv(G_meas, S_meas, G_irf, S_irf)
    G_theor, S_theor = phg.pure_phasor(tau, T_laser, harm)
    
    curve_irf = hv.Curve((time_bins, irf), "Time (ns)", "Counts (log)", label = "IRF").opts(color = "grey", line_dash = "dashed")
    curve_signal = hv.Curve((time_bins, noisy_signal), "Time (ns)", "Counts", label = "Signal").opts(color = "forestgreen", line_width = 3)
    plot_decay = (curve_irf * curve_signal).opts(title = f"Single exp (τ = {tau:.2f} ns)", width = 650, height = 550, 
                                                legend_position = 'top_right', logy = True, ylim = (1, None))
    
    points = [
        {'G': G_theor, 'S': S_theor, 'label': f'Target ({tau:.2f} ns)', 'color': 'lightskyblue', 'marker': 'o', 'markersize': 13, 'facecolor': 'none'},
        {'G': G_irf, 'S': S_irf, 'label': 'IRF', 'color': 'red', 'marker': 'x', 'markersize': 13, 'linewidth': 2.5 },
        {'G': G_meas, 'S': S_meas, 'label': 'Measured (Signal+IRF)', 'color': 'forestgreen', 'marker': 'o', 'markersize': 7},
        {'G': G_dec, 'S': S_dec, 'label': 'Final (Deconvolved)', 'color': 'steelblue', 'marker': '*', 'markersize': 12},
    ]
    lines = [{'G': [G_irf, G_meas], 'S': [S_irf, S_meas], 'color': 'gray', 'ls': ':'}]
    
    plot_phasor = phg.create_phasor_plot(T_laser, points, lines)
    return pn.Row(plot_decay, plot_phasor)



@pn.depends(
    t_laser_input, bins_input, counts_input, harmonic_input,
    t2_tau1, t2_tau2, t2_f1, t2_f2, t2_irf_mu, t2_irf_sig, t2_add_noise, t2_snr
)

def tab2_view(T_laser, n_bins, counts, harm, tau1, tau2, f1, f2, mu, sig, add_noise, snr):

    time_bins, irf = phf.irf_generation(sig, mu, counts, T_laser, n_bins)
    
    decay1, decay2 = phg.generate_decay(time_bins, tau1), phg.generate_decay(time_bins, tau2)
    mixed_decay = f1 * decay1 + f2 * decay2
    
    c1, _ = phf.simulation_with_IRF(decay1, irf, counts * f1, False, 0)
    c2, _ = phf.simulation_with_IRF(decay2, irf, counts * f2, False, 0)
    convol, noisy_signal = phf.simulation_with_IRF(mixed_decay, irf, counts, add_noise, snr)
    
    G1, S1 = phg.pure_phasor(tau1, T_laser, harm)
    G2, S2 = phg.pure_phasor(tau2, T_laser, harm)
    
    G_tm, S_tm = phf.phasor_calc_mix([G1, G2], [S1, S2], [f1, f2])
    
    G_meas, S_meas = phf.phasor_calc(noisy_signal, harm)
    G_irf, S_irf = phf.phasor_calc(irf, harm)
    G_dec, S_dec = phf.phasor_deconv(G_meas, S_meas, G_irf, S_irf)
    
    curve_irf = hv.Curve((time_bins, irf), "Tims (ns)", "Counts (log)", label = "IRF").opts(color = "grey", line_dash = "dashed")
    curve1 = hv.Curve((time_bins, c1), label = f"Exp1 ({tau1:.2f} ns)").opts(color = "orchid", alpha = 0.6, line_width = 2)
    curve2 = hv.Curve((time_bins, c2), label = f"Exp2 ({tau2:.2f} ns)").opts(color = "darkorchid", alpha = 0.6, line_width = 2)
    curve_signal = hv.Curve((time_bins, noisy_signal), label = "Mixed signal").opts(color = "forestgreen", line_width = 3)
    plot_decay = (curve_irf * curve1 * curve2 * curve_signal).opts(title =f"Double exp (f1 = {f1:.2f}, f2 = {f2:.2f})", 
                                                                   width = 650, height = 550, legend_position = 'top_right', logy = True, ylim = (1, None))
    
    points = [
        {'G': G1, 'S': S1, 'label': f'Exp1 ({tau1:.2f} ns)', 'color': 'orchid', 'marker': 'o', 'markersize': 8},
        {'G': G2, 'S': S2, 'label': f'Exp2 ({tau2:.2f} ns)', 'color': 'darkorchid', 'marker': 'o', 'markersize': 8},
        {'G': G_tm, 'S': S_tm, 'label': 'Target Mix', 'color': 'lightskyblue', 'marker': 'o', 'markersize': 13, 'facecolor': 'none'},
        {'G': G_irf, 'S': S_irf, 'label': 'IRF', 'color': 'red', 'marker': 'x', 'markersize': 13, 'linewidth': 2.5},
        {'G': G_meas, 'S': S_meas, 'label': 'Measured (Mix+IRF)', 'color': 'forestgreen', 'marker': 'o', 'markersize': 7},
        {'G': G_dec, 'S': S_dec, 'label': 'Final', 'color': 'steelblue', 'marker': '*', 'markersize': 12},
    ]
    lines = [
        {'G': [G1, G2], 'S': [S1, S2], 'color': 'gray', 'ls': '--'},
        {'G': [G_irf, G_meas], 'S': [S_irf, S_meas], 'color': 'gray', 'ls': ':'}
    ]
    
    plot_phasor = phg.create_phasor_plot(T_laser, points, lines)
    return pn.Row(plot_decay, plot_phasor)




@pn.depends(
    t_laser_input, bins_input, counts_input, harmonic_input,
    t3_tau1, t3_tau2, t3_tau3, t3_f1, t3_f2, t3_f3, t3_irf_mu, t3_irf_sig, t3_add_noise, t3_snr
)
def tab3_view(T_laser, n_bins, counts, harm, tau1, tau2, tau3, f1, f2, f3, mu, sig, add_noise, snr):

    time_bins, irf = phf.irf_generation(sig, mu, counts, T_laser, n_bins)
    
    decay1, decay2, decay3 = phg.generate_decay(time_bins, tau1), phg.generate_decay(time_bins, tau2), phg.generate_decay(time_bins, tau3)
    mixed_decay = f1 * decay1 + f2 * decay2 + f3 * decay3
    
    c1, _ = phf.simulation_with_IRF(decay1, irf, counts * f1, False, 0)
    c2, _ = phf.simulation_with_IRF(decay2, irf, counts * f2, False, 0)
    c3, _ = phf.simulation_with_IRF(decay3, irf, counts * f3, False, 0)
    convol, noisy_signal = phf.simulation_with_IRF(mixed_decay, irf, counts, add_noise, snr)
    
    G1, S1 = phg.pure_phasor(tau1, T_laser, harm)
    G2, S2 = phg.pure_phasor(tau2, T_laser, harm)
    G3, S3 = phg.pure_phasor(tau3, T_laser, harm)
    
    G_tm, S_tm = phf.phasor_calc_mix([G1, G2, G3], [S1, S2, S3], [f1, f2, f3])
    
    G_meas, S_meas = phf.phasor_calc(noisy_signal, harm)
    G_irf, S_irf = phf.phasor_calc(irf, harm)
    G_dec, S_dec = phf.phasor_deconv(G_meas, S_meas, G_irf, S_irf)
    
    curve_irf = hv.Curve((time_bins, irf), "Time (ns)", "Counts (log)", label = "IRF").opts(color = "grey", line_dash = "dashed")
    curve1 = hv.Curve((time_bins, c1), label = f"Exp1 ({tau1:.2f} ns)").opts(color = "orchid", alpha = 0.6, line_width = 2)
    curve2 = hv.Curve((time_bins, c2), label = f"Exp2 ({tau2:.2f} ns)").opts(color = "darkorchid", alpha = 0.6, line_width = 2)
    curve3 = hv.Curve((time_bins, c3), label = f"Exp3 ({tau3:.2f} ns)").opts(color = "orange", alpha = 0.6, line_width = 2)
    curve_signal = hv.Curve((time_bins, noisy_signal), label = "Mixed signal").opts(color = "forestgreen", line_width = 3)
    plot_decay = (curve_irf * curve1 * curve2 * curve3 * curve_signal).opts(title = f"Triple exp (f1 = {f1:.2f}, f2 = {f2:.2f}, f3 = {f3:.2f})", 
                                                                            width = 650, height = 550, legend_position = 'top_right', logy = True, ylim = (1, None))
    # phasor plane territory

    points = [
        {'G': G1, 'S': S1, 'label': f'Exp1 ({tau1:.2f} ns)', 'color': 'orchid', 'marker': 'o', 'markersize': 8},
        {'G': G2, 'S': S2, 'label': f'Exp2 ({tau2:.2f} ns)', 'color': 'darkorchid', 'marker': 'o', 'markersize': 8},
        {'G': G3, 'S': S3, 'label': f'Exp3 ({tau3:.2f} ns)', 'color': 'orange', 'marker': 'o', 'markersize': 8},
        {'G': G_tm, 'S': S_tm, 'label': 'Target Mix', 'color': 'lightskyblue', 'marker': 'o', 'markersize': 13, 'facecolor': 'none'},
        {'G': G_irf, 'S': S_irf, 'label': 'IRF', 'color': 'red', 'marker': 'x', 'markersize': 13, 'linewidth': 2.5},
        {'G': G_meas, 'S': S_meas, 'label': 'Measured', 'color': 'forestgreen', 'marker': 'o', 'markersize': 7},
        {'G': G_dec, 'S': S_dec, 'label': 'Final', 'color': 'steelblue', 'marker': '*', 'markersize': 12},
    ]
    lines = [
        {'G': [G1, G2, G3, G1], 'S': [S1, S2, S3, S1], 'color': 'gray', 'ls': '--'},
        {'G': [G_irf, G_meas], 'S': [S_irf, S_meas], 'color': 'gray', 'ls': ':'} 
    ]
    
    plot_phasor = phg.create_phasor_plot(T_laser, points, lines)
    return pn.Row(plot_decay, plot_phasor)



# ─── Sidebar Layout ────────────────────────────────────────────────────────────


global_card = pn.Card(
    t_laser_input, bins_input, counts_input, harmonic_input,
    title = "Parametri Globali", collapsed=False,
)
tab1_card = pn.Card(t1_tau, t1_irf_mu, t1_irf_sig, t1_add_noise, t1_snr, title="1) Single exponential", collapsed = True)
tab2_card = pn.Card(t2_tau1, t2_tau2, t2_f1, t2_f2, t2_irf_mu, t2_irf_sig, t2_add_noise, t2_snr, title = "2) double exponential", collapsed = True)
tab3_card = pn.Card(t3_tau1, t3_tau2, t3_tau3, t3_f1, t3_f2, t3_f3, t3_irf_mu, t3_irf_sig, t3_add_noise, t3_snr, title = "3) Triple exponential", collapsed = True)
sidebar = pn.Column(global_card, tab1_card, tab2_card, tab3_card, tab4_card_ext, margin = (10, 10))




# ─── Main Layout ───────────────────────────────────────────────────────────────

header = pn.Column(
    pn.pane.Markdown("# FLIM phasors analysis"),
    pn.pane.Markdown("Select a tab to explore different modalities. Global parameters apply to all tabs"),
)
main_tabs = pn.Tabs(
    ("1) Single exponential", tab1_view),
    ("2) Double exponential", tab2_view),
    ("3) Triple exponential", tab3_view),
    ("4) Image analysis", tab4_view_ext),

)
main_area = pn.Column(header, main_tabs)



# ── SYNC SIDEBAR (left) AND TABS (top) ─────────────────────────────────────────

_tab_cards = {0: tab1_card, 1: tab2_card, 2: tab3_card, 3: tab4_card_ext}
_syncing = False

def _sync_cards(event):
    global _syncing
    if _syncing: return
    _syncing = True
    try:
        for idx, card in _tab_cards.items():
            card.collapsed = (idx != event.new)
    finally:
        _syncing = False

def _sync_tab_from_card(card_idx, event):
    global _syncing
    if _syncing or event.new: return
    _syncing = True
    try:
        main_tabs.active = card_idx
        for idx, card in _tab_cards.items():
            if idx != card_idx:
                card.collapsed = True
    finally:
        _syncing = False

main_tabs.param.watch(_sync_cards, "active")
for _card_idx, _card in _tab_cards.items():
    _card.param.watch(lambda e, i = _card_idx: _sync_tab_from_card(i, e), "collapsed")

# ─── START ────────────────────────────────────────────────────────

template = pn.template.MaterialTemplate(
    site = "FLIM Analysis ",
    title = "Phasor App",
    sidebar = [sidebar],
    main = [main_area],
)

template.servable()

if __name__ == "__main__":
    pn.serve(template, show = True, port = 5900)