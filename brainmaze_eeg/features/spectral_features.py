# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy.stats as stats


def mean_frequency(args):
    """
    Calculate the mean dominant frequency of the power spectral density.
    
    **Big Picture:**
    This function computes the centroid (weighted average) of the frequency spectrum,
    which indicates where the "center of mass" of the spectral power lies. This is
    particularly useful in sleep stage classification and EEG analysis to characterize
    the dominant frequency content of brain activity.
    
    **Technical Details:**
    The mean frequency is calculated as the power-weighted average frequency:
    
    .. math::
        f_{mean} = \\frac{\\sum_{i} P(f_i) \\cdot f_i}{\\sum_{i} P(f_i)}
    
    where P(f_i) is the power spectral density at frequency f_i. The calculation
    is performed over the frequency range defined by the minimum and maximum
    of the provided frequency bands.
    
    **Use Case:**
    Commonly used to distinguish between sleep stages where different frequency
    bands dominate (e.g., delta in deep sleep vs. alpha/beta in wakefulness).
    
    Parameters
    ----------
    args : dictionary
        - 'psd' (*numpy.ndarray[n_samples, n_freq_samples]*) - one-sided PSD
        - 'fbands' (*list of lists*) - frequency bands in which the feature is to be calculated [[0.5, 4], [5, 9]]
        - 'freq' (*numpy.array[n_freq_samples]*) - reference frequency array for the PSD

    Returns
    -------
    x : list(numpy.array)
        Calculated mean dominant frequency for each sample
    feature_name : list(str)
        Feature name: 'MEAN_DOMINANT_FREQUENCY'
        
    References
    ----------
    MATLAB meanfreq: https://www.mathworks.com/help/signal/ref/meanfreq.html

    """
    Pxx = args['psd']
    bands = args['fbands']
    freq = args['freq']

    f = args.freq

    min_position = np.nanargmin(np.abs(f - bands.min()))
    max_position = np.nanargmin(np.abs(f - bands.max()))

    P = Pxx[:, min_position: max_position + 1]
    f = f[min_position: max_position + 1]

    f = np.reshape(f, (1, -1))
    pwr = np.sum(P, axis=1)
    mnfreq = np.dot(P, f.T).squeeze() / pwr
    return [mnfreq], ['MEAN_DOMINANT_FREQUENCY']


def median_frequency(args):
    """
    Calculate the spectral median frequency of the power spectral density.
    
    **Big Picture:**
    The median frequency divides the power spectrum into two equal halves of power.
    It is the frequency below which 50% of the total power is contained. This metric
    is robust to outliers and provides an alternative to mean frequency for characterizing
    the spectral content of signals, especially useful when the power distribution is skewed.
    
    **Technical Details:**
    The median frequency f_median is defined as the frequency where the cumulative
    power reaches 50% of the total power:
    
    .. math::
        \\int_0^{f_{median}} P(f) df = 0.5 \\cdot \\int_0^{f_{max}} P(f) df
    
    The calculation is performed over the frequency range defined by the minimum
    and maximum of the provided frequency bands. This implementation uses cumulative
    sum and finds the crossover point where cumulative power exceeds half the total.
    
    **Use Case:**
    Often preferred over mean frequency in EEG analysis when dealing with non-Gaussian
    spectral distributions or when the presence of high-frequency noise might skew
    the mean frequency estimate.
    
    Parameters
    ----------
    args : dictionary
        - 'psd' (*numpy.ndarray[n_samples, n_freq_samples]*) - one-sided PSD
        - 'fbands' (*list of lists*) - frequency bands in which the feature is to be calculated [[0.5, 4], [5, 9]]
        - 'freq' (*numpy.array[n_freq_samples]*) - reference frequency array for the PSD

    Returns
    -------
    x : list(numpy.array)
        Calculated spectral median frequency for each sample
    feature_name : list(str)
        Feature name: 'SPECTRAL_MEDIAN_FREQUENCY'
        
    References
    ----------
    MATLAB medfreq: https://www.mathworks.com/help/signal/ref/medfreq.html

    """
    Pxx = args['psd']
    bands = args['fbands']
    freq = args['freq']


    pwr = np.sum(Pxx, axis=1)
    #f = 0.5 * fs * np.arange(1, Pxx.shape[1]) / Pxx.shape[1]
    f = args.freq
    min_position = np.nanargmin(np.abs(f - bands.min()))
    max_position = np.nanargmin(np.abs(f - bands.max()))

    P = Pxx[:, min_position: max_position + 1]
    f = f[min_position: max_position + 1]

    pwr05 = np.repeat(pwr / 2, P.shape[1]).reshape(P.shape)
    P = np.cumsum(np.abs(P), axis=1)

    medfreq_pos = np.argmax(np.diff(P > pwr05, axis=1), axis=1) + 1
    medfreq = f.squeeze()[medfreq_pos]
    return [medfreq], ['SPECTRAL_MEDIAN_FREQUENCY']


def mean_bands(args):
    """
    Calculate mean power spectral density for each frequency band.
    
    **Big Picture:**
    This function computes the average spectral power within specific frequency bands
    (e.g., delta: 0.5-4 Hz, theta: 4-8 Hz, alpha: 8-12 Hz). These band-specific power
    measurements are fundamental features in EEG analysis and sleep stage classification,
    as different brain states are characterized by distinct patterns of oscillatory
    activity in different frequency ranges.
    
    **Technical Details:**
    For each frequency band [f_low, f_high], the mean PSD is calculated as:
    
    .. math::
        P_{band} = \\frac{1}{N} \\sum_{f_i \\in [f_{low}, f_{high}]} P(f_i)
    
    where N is the number of frequency bins in the band. This provides an absolute
    measure of power in each band, useful for comparing spectral characteristics
    across different signals or time segments.
    
    **Use Case:**
    Essential for sleep stage classification where:
    - Delta band (0.5-4 Hz) power increases in deep sleep (N3)
    - Alpha band (8-12 Hz) power is prominent during relaxed wakefulness
    - Beta band (12-30 Hz) power increases during active wakefulness
    - Theta band (4-8 Hz) power is characteristic of light sleep (N1, N2) and REM
    
    Parameters
    ----------
    args : dictionary
        - 'psd' (*numpy.ndarray[n_samples, n_freq_samples]*) - one-sided PSD
        - 'fbands' (*list of lists*) - frequency bands in which the feature is to be calculated [[0.5, 4], [5, 9]]
        - 'freq' (*numpy.array[n_freq_samples]*) - reference frequency array for the PSD

    Returns
    -------
    x : list(numpy.array)
        List of mean PSD values for each frequency band, one array per band
    feature_name : list(str)
        Feature names in format 'MEAN_PSD{low}-{high}Hz' with the frequencies
        concatenated directly (e.g., 'MEAN_PSD0.5-4Hz') for each band

    """
    Pxx = args['psd']
    bands = args['fbands']
    freq = args['freq']


    outp_params = []
    outp_msg = []
    for band in bands:
        subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
        outp_params.append(
            np.nanmean(subpsdx, axis=1)
        )
        outp_msg.append('MEAN_PSD' + str(band[0]) + '-' + str(band[1]) + 'Hz')
    return outp_params, outp_msg


def relative_bands(args):
    """
    Calculate relative power spectral density for each frequency band.
    
    **Big Picture:**
    This function computes the proportion of total power contained in each frequency
    band, normalizing by the total power across all bands. Relative power features
    are more robust to inter-individual differences in overall signal amplitude and
    better reflect the relative contribution of each frequency band to the overall
    brain activity. This normalization is crucial for reliable sleep stage classification
    and cross-subject comparisons.
    
    **Technical Details:**
    For each frequency band [f_low, f_high], the relative PSD is calculated as:
    
    .. math::
        P_{rel,band} = \\frac{\\sum_{f_i \\in [f_{low}, f_{high}]} P(f_i)}{\\sum_{f_i \\in [f_{min}, f_{max}]} P(f_i)}
    
    where [f_min, f_max] is the continuous frequency interval from the smallest lower
    bound to the largest upper bound across all provided bands (i.e., ``f_min = bands.min()``
    and ``f_max = bands.max()``). The sum of all relative band powers equals 1.0, making
    them interpretable as percentages of total power.
    
    **Use Case:**
    Preferred over absolute band powers for machine learning-based sleep staging because:
    - Accounts for individual differences in EEG amplitude
    - Reduces the impact of recording conditions (electrode impedance, amplifier gain)
    - Focuses on the spectral shape rather than absolute magnitude
    - More stable features across different recording sessions
    
    **Clinical Applications:**
    - Sleep stage classification: High relative delta in N3, high relative alpha in wake
    - Anesthesia depth monitoring: Shifts in relative power distributions
    - Cognitive state assessment: Changes in relative alpha/beta ratios
    
    Parameters
    ----------
    args : dictionary
        - 'psd' (*numpy.ndarray[n_samples, n_freq_samples]*) - one-sided PSD
        - 'fbands' (*list of lists*) - frequency bands in which the feature is to be calculated [[0.5, 4], [5, 9]]
        - 'freq' (*numpy.array[n_freq_samples]*) - reference frequency array for the PSD

    Returns
    -------
    x : list(numpy.array)
        List of relative PSD values for each frequency band (values between 0 and 1)
    feature_name : list(str)
        Feature names in format 'REL_PSD_{low}-{high}Hz' (e.g., 'REL_PSD_0.5-4Hz') for each band

    """
    Pxx = args['psd']
    bands = args['fbands']
    freq = args['freq']


    outp_params = []
    outp_msg = []

    fullpsdx = np.nansum(Pxx[:, (freq >= bands.min()) & (freq <= bands.max())], axis=1)
    for band in bands:
        subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
        outp_params.append(
            np.nansum(subpsdx, axis=1) / fullpsdx
        )
        outp_msg.append('REL_PSD_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
    return outp_params, outp_msg


def normalized_entropy(args):
    """
    **Spectral entropy (Shannon Entropy)**

    - Estimates Shannon Entropy of a spectrum on a frequency range defined as min-to-max of frequency bands at the input.
    - Source: https://www.mathworks.com/help/wavelet/ref/wentropy.html

    Parameters
    ----------
    args : dictionary
        - 'psd' (*numpy.ndarray[n_samples, n_freq_samples]*) - one-sided PSD
        - 'fbands' (*list of lists*) - frequency bands in which the feature is to be calculated [[0.5, 4], [5, 9]]
        - 'freq' (*numpy.array[n_freq_samples]*) - reference frequency array for the PSD

    Returns
    -------
    x : list(numpy.array)
        Calculated features for individual frequency bands
    feature_name : list(numpy.array)
        Feature names

    """
    Pxx = args['psd']
    bands = args['fbands']
    freq = args['freq']

    subpsdx = Pxx[:, (freq >= bands.min()) & (freq <= bands.max())]
    return [
               stats.entropy(subpsdx ** 2, axis=1)
           ], [
               'SPECTRAL_ENTROPY_' + str(bands.min()) + '-' + str(bands.max()) + 'Hz'
           ]


def non_normalized_entropy(args):
    """
    **Spectral entropy (Shannon Entropy)**

    - Estimates Shannon Entropy of a spectrum on a frequency range defined as min-to-max of frequency bands at the input.
    - Source: https://www.mathworks.com/help/wavelet/ref/wentropy.html

    Parameters
    ----------
    args : dictionary
        - 'psd' (*numpy.ndarray[n_samples, n_freq_samples]*) - one-sided PSD
        - 'fbands' (*list of lists*) - frequency bands in which the feature is to be calculated [[0.5, 4], [5, 9]]
        - 'freq' (*numpy.array[n_freq_samples]*) - reference frequency array for the PSD

    Returns
    -------
    x : list(numpy.array)
        Calculated features for individual frequency bands
    feature_name : list(numpy.array)
        Feature names

    """
    Pxx = args['psd']
    bands = args['fbands']
    freq = args['freq']

    subpsdx = Pxx[:, (freq >= bands.min()) & (freq <= bands.max())]
    return [
               - np.sum(subpsdx ** 2 * np.log(subpsdx ** 2), axis=1)
           ], [
               'SPECTRAL_ENTROPY_' + str(bands.min()) + '-' + str(bands.max()) + 'Hz'
           ]


def normalized_entropy_bands(args):
    """
    **Spectral entropy (Shannon Entropy)**

    - Estimates Shannon Entropy of a spectrum on a frequency range defined as min-to-max of frequency bands at the input.
    - Source: https://www.mathworks.com/help/wavelet/ref/wentropy.html

    Parameters
    ----------
    args : dictionary
        - 'psd' (*numpy.ndarray[n_samples, n_freq_samples]*) - one-sided PSD
        - 'fbands' (*list of lists*) - frequency bands in which the feature is to be calculated [[0.5, 4], [5, 9]]
        - 'freq' (*numpy.array[n_freq_samples]*) - reference frequency array for the PSD

    Returns
    -------
    x : list(numpy.array)
        Calculated features for individual frequency bands
    feature_name : list(numpy.array)
        Feature names

    """
    Pxx = args['psd']
    bands = args['fbands']
    freq = args['freq']


    outp_params = []
    outp_msg = []
    for band in bands:
        subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
        outp_params.append(
            stats.entropy(subpsdx ** 2, axis=1)
        )
        outp_msg.append('SPECTRAL_ENTROPY_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
    return outp_params, outp_msg


def non_normalized_entropy_bands(args):
    """
    **Spectral entropy (Shannon Entropy)**

    - Estimates Shannon Entropy of a spectrum on a frequency range defined as min-to-max of frequency bands at the input.
    - Source: https://www.mathworks.com/help/wavelet/ref/wentropy.html

    Parameters
    ----------
    args : dictionary
        - 'psd' (*numpy.ndarray[n_samples, n_freq_samples]*) - one-sided PSD
        - 'fbands' (*list of lists*) - frequency bands in which the feature is to be calculated [[0.5, 4], [5, 9]]
        - 'freq' (*numpy.array[n_freq_samples]*) - reference frequency array for the PSD

    Returns
    -------
    x : list(numpy.array)
        Calculated features for individual frequency bands
    feature_name : list(numpy.array)
        Feature names

    """
    Pxx = args['psd']
    bands = args['fbands']
    freq = args['freq']


    outp_params = []
    outp_msg = []
    for band in bands:
        subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
        outp_params.append(
            - np.sum(subpsdx ** 2 * np.log(subpsdx ** 2), axis=1)
        )
        outp_msg.append('SPECTRAL_ENTROPY_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
    return outp_params, outp_msg



