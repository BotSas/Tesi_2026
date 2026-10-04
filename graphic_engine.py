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




# __ Initialization __________________________________________________________ 

#_DATA_DIR = Path("data")

pn.extension(
    sizing_mode="stretch_width",
    loading_indicator=True,
    loading_spinner="dots",
)
hv.extension("bokeh")





# __ Plotting functions __________________________________________________________ 


def create_phasor_plot(ds: xr.Dataset):
    df_samples = ds[["G", "S"]].to_dataframe().reset_index()
    df_ref = ds[["G_ref", "S_ref"]].to_dataframe().reset_index()

    ref_chart = df_ref.hvplot.points(
        x="G_ref",
        y="S_ref",
        color="wavelength_ref",
        cmap="spectral_r",
        size=28,
        colorbar=False,
        tools=["hover"],
    )

    single_sample = len(df_samples) == 1
    if single_sample:
        color_mapping = {int(df_samples["sample"].iloc[0]): "steelblue"}
        sample_chart = df_samples.hvplot.points(
            x="G",
            y="S",
            color="steelblue",
            size=50,
            colorbar=False,
            tools=["hover", "box_select", "lasso_select", "tap"],
        )
    else:
        color_mapping = dict(
            zip(df_samples["sample"].tolist(), cycle(colorcet.b_glasbey_hv))
        )
        sample_chart = df_samples.hvplot.points(
            x="G",
            y="S",
            color="sample",
            cmap=color_mapping,
            size=45,
            colorbar=False,
            tools=["hover", "box_select", "lasso_select", "tap"],
        )

    phasor_plot = (ref_chart * sample_chart).opts(
        hv.opts.Points(
            frame_width=500,
            frame_height=500,
            padding=0.1,
            xlabel="G",
            ylabel="S",
            show_grid=True,
            show_legend=False,
            title="Phasor Plot",
        )
    )
    return phasor_plot, sample_chart, color_mapping






# ── Global widgets ────────────────────────────────────────────────────────────

T_laser_input = pn.widgets.FloatInput(
    name = "Laser period (ns)", value = 12, step = 0.1, width = 12
)
N_evt_input_ = pn.widgets.FloatInput(
    name = "Number of events", value = 10, step=10, width=10000
)
tau1_input = pn.widgets.FloatInput(
    name="Lifetime of fluophore n. 1 (ns)", value = 4, step=0.1, width=15
)
harmonic_input = pn.widgets.IntInput(
    name="Harmonic N", value=1, step=1, start=1, width=160
)
show_individual = pn.widgets.Toggle(
    name="Show Individual Spectra", button_type="default", value=False
)

# ── Panel app ────────────────────────────────────────────────────────────

app = pn.Column(
    "# Phasor Analysis",
    pn.Row(
        T_laser_input,
        N_evt_input_,
        tau1_input,
        harmonic_input,
    ),
    show_individual,
)

app.servable()