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



# BINNED NLL FUNCTION

def expon_cdf(x, A, τ):
    return A * τ * (1.0 - np.exp(-x / τ))   # I need cdf for the binned likelihood 


#######################################################################################################
#######################################################################################################
#######################################################################################################


# PHASOR PLOT AND NOISE ADDITION


def phasor_calc (n, harm_n = 1):   # given a histogram n (xe), calculates the phasor coordinates G and S

    fft_vals = np.fft.rfft(n)
    fft_norm = fft_vals / fft_vals[0]
    
    G, S = np.real(fft_norm[harm_n]), -np.imag(fft_norm[harm_n])

    return G, S




###################################################################################################



def add_noise(counts, snr_db):  # Adds GAUSSIAN NOISE (why not Poisson? Need to check)

    rng = np.random.default_rng()   
    snr_linear = 10 ** (snr_db / 10)  # convert SNR from dB to linear scale 

    sig_power = np.mean(counts ** 2, axis = 0) 
    noise_std = np.sqrt(sig_power / snr_linear)
    noise = rng.standard_normal(counts.shape) * noise_std
    
    return np.clip(counts + noise, 0, None)




###################################################################################################


def phasor_calc_mix(G_array, S_array, fractions):

    G_array = np.asarray(G_array)
    S_array = np.asarray(S_array)
    fractions = np.asarray(fractions, dtype=float)

    fractions /= np.sum(fractions)  # safety feature
    G_mix, S_mix = np.sum(fractions * G_array), np.sum(fractions * S_array)

    return G_mix, S_mix



###################################################################################################




def projection_length(point, A, B):   # asarray converts points/lines (Pandas dataframes) to a numpy array
    
    """
    Projects a point onto a line segment and returns the segment lengths.
    """

    P = np.asarray(point, dtype = float)      # points that is to be projected  
    A, B = np.asarray(line_start, dtype = float), np.asarray(line_end, dtype = float)   # extremes of the full segment (formed by tau of the 2 exponential decays)

    line_vector = B - A
    H_factor = np.dot(P - A, line_vector) / np.dot(line_vector, line_vector)  # valued 0 to 1 (how far from A on the segment H is), H is projected point
    H = A + H_factor * line_vector

    segmA, segmB = np.linalg.norm(H - A), np.linalg.norm(B - H)       # segments lengths

    return segmA, segmB



###################################################################################################



def projection_area(point, A, B, C):   # asarray converts points/lines (Pandas dataframes) to a numpy array

    """
    Works the same as the previous function but for 3 exponential decays. Calculates areas on the phasor plane corresponding to each fluorophor
    """

    point, A, B, C = map(np.asarray, (point, A, B, C))
    big_triangle = triangle_area(A, B, C)

    return (
        triangle_area(point, B, C) / big_triangle,
        triangle_area(point, C, A) / big_triangle,
        triangle_area(point, A, B) / big_triangle,
    )


###################################################################################################



def triangle_area(p1, p2, p3):              # self explainatory

    p1, p2, p3 = map(np.asarray, (p1, p2, p3))

    return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))



#######################################################################################################
#######################################################################################################
#######################################################################################################




# SIMULATION OF DATA + IRF DECONVOLUTION

def irf_generation(σ, μ, counts, T_laser, num_bins = 30):       # gaussian IRF

    time_bins = np.linspace(0, T_laser, num_bins)
    
    irf = np.exp(-((time_bins - μ)**2) / (2 * σ**2))
    
    irf /= np.sum(irf)
    irf *= counts
    
    return time_bins, irf



#############################################################################################################



def simulation_with_IRF(exp1data1, irf, counts, noise_flag, snr_db): 

    n_bins = len(irf)
    
    # Use a circular convolution so the signal stays consistent with the laser period.
    convol = np.fft.ifft(np.fft.fft(exp1data1) * np.fft.fft(irf)).real      # Here I'm using the phasor method instead of 
                                                                            # np.convolve (slow + doesn't handle periodicity)

    convol = np.clip(convol, 0, None)
    convol = (convol / np.sum(convol)) * counts  # Normalization
    
    #NOISEEEEE
    if noise_flag: noisy_signal = add_noise(convol, snr_db) 
    else: noisy_signal = convol
    
    return convol, noisy_signal




##############################################################################################################




def phasor_deconv(G_meas, S_meas, G_irf, S_irf): 
 
    z_meas = G_meas + 1j * S_meas       # imaginary unit i = 1j
    z_irf = G_irf + 1j * S_irf
    
    z_deconvol = z_meas / z_irf       # in phasor plane, convolution = moltiplication
    
    G_deconvol, S_deconvol = np.real(z_deconvol), np.imag(z_deconvol)
    
    return G_deconvol, S_deconvol





#######################################################################################################
#######################################################################################################
#######################################################################################################



# IMPORTED IMAGE + MEDIAN FILTER

def extract_irf(calib_image, τ_fluor, T_laser): 

    ''' Given a calibration image with a known fluorescence lifetime, extracts the IRF phasor coordinates '''  

    tot_calib = calib_image.sum(dim = ("x", "y"))
    G_ref, S_ref = phasor_calc(tot_calib) 

    ω = 2 * np.pi / T_laser
    ωτ = ω * τ_fluor
    
    G_true = 1 / (1 + ωτ**2)
    S_true = ωτ / (1 + ωτ**2)
    G_irf, S_irf = phasor_deconv(G_ref, S_ref, G_true, S_true)
    
    return G_irf, S_irf



##############################################################################################################




def image_phasor_calc(image, harm = 1):

    '''Calculates the phasor coordinates for each pixel in a [t, x, y] image.'''

    fft_result = xr.DataArray(
        np.fft.rfft(image.values, axis=0),
        dims = ("frequency", "x", "y"),
        coords = {
            "x": image.coords["x"],
            "y": image.coords["y"],
        },
    )

    fft0 = fft_result.isel(frequency=0).real            # fft0 = DC = intensity = total photons per pixel
    fft0 = fft0.where(fft0 != 0, 1e-10)                 # crash fix - "fills" 0 values (empty bins) with very low count

    G_img = fft_result.isel(frequency=harm).real / fft0
    S_img = -fft_result.isel(frequency=harm).imag / fft0

    return G_img, S_img, fft0



##############################################################################################################




def filter_image_phasors(image, threshold_pct=0.05):

    ''' Mask for bg noise: higher_threshold = higher noise '''

    G_img, S_img, fft0 = image_phasor_calc(image)

    count_peak = fft0.max().item()
    threshold = count_peak * threshold_pct

    mask = fft0 > threshold
    
    print(f"Max count: {count_peak:.0f} γ. Threshold: {threshold:.0f} γ.")  
    print(f"Pixels over threshold: {int(mask.sum())}/{mask.size}")

    G_valid = G_img.where(mask).stack(pixel = ("x", "y")).dropna("pixel")     
    S_valid = S_img.where(mask).stack(pixel = ("x", "y")).dropna("pixel")

    return G_valid, S_valid



##############################################################################################################



def cloud_data(df_phasors):

    """
    Calculates average point and standard deviation of a cloud of data
    """

    return df_phasors.mean(), df_phasors.std()    # dataset of G and S combined



##############################################################################################################


def phasors_from_region(image, roi_mask, median_size , threshold_pct):    # ROI = region of interest

    '''
    Calculates G and S from  spatial region of interest (can be whole image). Takes as input:
    - image as a xr.DataArray
    - roi_mask: 2D boolean array (x, y)
    '''

    G_img, S_img, fft0 = image_phasor_calc(image)   # phasors relative to the whole image
    
    fft0_med, G_med, S_med = phasor_filter_median(      # median filter on each point in phasor plane
        fft0.values, 
        G_img.values, 
        S_img.values,
        size = median_size,
        repeat = 2
    )
    
    threshold_val = np.nanmax(fft0_med) * threshold_pct     # tresholding - every pixel below treshold is set to NaN
    fft0_thresh, G_thresh, S_thresh = phasor_threshold(
        fft0_med,
        G_med,
        S_med,
        mean_min = threshold_val
    )
    
    valid_mask = roi_mask & ~np.isnan(G_thresh)     # only keeps pixel which are not NaN and inside ROI
    
    df_roi_phasors = pd.DataFrame({
        "G": G_thresh[valid_mask],
        "S": S_thresh[valid_mask]
    })
    
    return df_roi_phasors
    

