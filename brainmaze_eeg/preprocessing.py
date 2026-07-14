
import numpy as np
import scipy.signal as signal
from typing import Tuple
from scipy.ndimage import binary_dilation

from brainmaze_utils.signal import PSD, buffer


def channel_data_rate_thresholding(x: np.typing.NDArray[np.float64], threshold_data_rate: float=0.1):
    """
    Masks the whole channel [nchans, nsamples] with nans if the channel data rate is below the threshold.

    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.
        dr_threshold (float, optional): Drop rate threshold for masking. Default is 0.1.

    Returns:
        np.ndarray: Signal with masked values replaced by NaNs.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """

    ndim = x.ndim

    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]  # Add a new axis to make it 2D

    ch_mask = 1 - (np.isnan(x).sum(axis=1) / x.shape[1]) <= threshold_data_rate
    x[ch_mask, :] = np.nan

    if ndim == 1:
        x = x[0]

    return x


def replace_nans_with_median(x: np.typing.NDArray[np.float64]):
    """
    Replaces NaN values in the input signal with the median of the non-NaN values along each channel.

    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - processed_signal (np.ndarray): Signal with NaN values replaced by the median.
            - mask (np.ndarray): Boolean mask indicating the positions of NaN values in the original signal.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """

    ndim = x.ndim

    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]  # Add a new axis to make it 2D

    mask = np.isnan(x)

    if not mask.any(): # if no nans, just return
        if ndim == 1:
            x = x[0]
            mask = mask[0]

        return x, mask

    med_vals = np.nanmedian(x, axis=1, keepdims=True)
    x = np.where(mask, med_vals, x)

    if ndim == 1:
        x = x[0]

    return x, mask


def filter_powerline(x: np.typing.NDArray[np.float64], fs: float, frequency_powerline: float=60):
    """
    Remove powerline noise from EEG signals using a notch filter.
    
    **Big Picture:**
    Powerline interference (50 Hz in Europe, 60 Hz in North America) is one of the most
    common artifacts in electrophysiological recordings. This narrow-band noise arises
    from electromagnetic coupling with the electrical grid and can obscure physiological
    signals. This function applies a notch filter to remove this interference while
    preserving the EEG signal content at other frequencies.
    
    **Technical Details:**
    The function uses an infinite impulse response (IIR) notch filter with:
    - Quality factor Q=10, providing a narrow rejection bandwidth
    - Zero-phase filtering (filtfilt) to avoid phase distortion
    - Automatic handling of NaN values by temporarily replacing them with median values
    
    The notch filter attenuates frequencies in a narrow band centered at the powerline
    frequency while minimally affecting adjacent frequencies. After filtering, NaN
    values are restored to their original locations to preserve data quality indicators.
    
    **Important Considerations:**
    - This approach may cause ringing artifacts around data gaps (NaN regions)
    - Alternative: Use robust filtering methods that handle missing data explicitly
    - The filter is applied independently to each channel in multi-channel recordings
    
    **Use Case:**
    Apply during preprocessing of raw EEG/iEEG recordings before feature extraction
    or sleep stage classification. Essential for clean spectral analysis and preventing
    powerline noise from being misinterpreted as physiological rhythms.
    
    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.
        fs (float): Sampling frequency in Hz.
        frequency_powerline (float): Powerline noise frequency (typically 50 or 60 Hz).

    Returns:
        np.ndarray: Signal filtered with notch filter, same shape as input.

    Raises:
        ValueError: If the input signal is not 1D or 2D.

    """

    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    mask = np.isnan(x)
    x = np.where(mask, np.nanmedian(x, axis=1, keepdims=True), x)  # substitute nans with median for notch filter

    b, a = signal.iirnotch(w0=frequency_powerline, Q=10, fs=fs)
    x = signal.filtfilt(b, a, x, axis=1)

    x[mask] = np.nan

    if ndim == 1:
        x = x[0]

    return x


def detect_powerline_segments(
        x: np.typing.NDArray[np.float64],
        fs: float,
        window_s: float = 0.5,
        powerline_freq:float = 60,
        threshold_ratio:float = 1000
):
    """
    Detect segments contaminated by powerline noise in EEG recordings.
    
    **Big Picture:**
    Powerline noise contamination can vary over time due to changes in electrode
    impedance, patient movement, or proximity to electrical equipment. Rather than
    filtering all data uniformly, this function identifies time segments with severe
    powerline contamination, allowing for selective processing or exclusion of
    problematic data. This approach preserves clean data segments while flagging
    contaminated ones for special handling.
    
    **Technical Details:**
    The detection algorithm:
    
    1. Divides the signal into short windows (default 0.5s)
    2. Computes power spectral density for each window
    3. Calculates reference power in physiological band (2-40 Hz)
    4. Measures power at powerline frequency and its harmonics (60, 120, 180 Hz, etc.)
    5. Flags segments where powerline power exceeds physiological power by threshold_ratio
    
    .. math::
        contaminated = \\frac{P_{powerline}}{P_{2-40Hz}} > threshold
    
    The default threshold of 1000 indicates that powerline power is 1000× greater
    than the average physiological power, suggesting severe contamination.
    
    **Harmonics Detection:**
    The function checks not just the fundamental powerline frequency but also its
    harmonics, as powerline interference often creates spectral peaks at multiples
    of the base frequency (60 Hz, 120 Hz, 180 Hz, etc.).
    
    **Use Case:**
    - Preprocessing pipeline for automatic data quality assessment
    - Adaptive filtering strategies (filter only contaminated segments)
    - Data quality reporting and visualization
    - Exclusion criteria for feature extraction in clean segments only
    
    **Practical Applications:**
    - Sleep studies: Exclude epochs with excessive powerline noise from analysis
    - Real-time monitoring: Alert when powerline noise exceeds acceptable levels
    - Research studies: Ensure consistent data quality across recordings
    
    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.
        fs (float): Sampling frequency in Hz.
        window_s (float): Length of the detection window in seconds. Default is 0.5 seconds.
        powerline_freq (float): Frequency of the powerline noise (50 or 60 Hz). Default is 60 Hz.
        threshold_ratio (float): Threshold ratio indicating how many times the power of the 
                                powerline noise is higher than average power in 2-40 Hz band. 
                                Default is 1000.

    Returns:
        np.ndarray: Boolean array indicating the presence of powerline noise for every detection window.
                   Shape is (n_channels, n_windows) or (n_windows,) for 1D input.

    """

    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    xb =  np.array([
        buffer(x_, fs, segm_size=window_s, drop=True) for x_ in x
    ])
    xb = xb - np.nanmean(xb, axis=2, keepdims=True)
    f, pxx = PSD(xb, fs)

    max_freq = f[-1]

    idx_lower_band = (f>=2) & (f <= 40)
    pow_40 = np.nanmean(pxx[:, :, idx_lower_band], axis=2, keepdims=True) # since we always buffer 1 second, we can use absolute indexes

    idx_pline = np.array([
        np.where((f >= f_det -2) & (f <= f_det + 2))[0] for f_det in np.arange(powerline_freq, max_freq, powerline_freq)
    ]).flatten()
    idx_pline = np.round(idx_pline).astype(np.int64)

    pow_pline = np.nanmax(pxx[:, :, idx_pline], axis=2, keepdims=True)

    pow_rat = pow_pline / pow_40

    pow_rat = pow_rat.squeeze(axis=2)
    detected_noise = pow_rat >= threshold_ratio

    if ndim == 1:
        detected_noise = detected_noise[0]

    return detected_noise


def detect_outlier_segments(
        x: np.typing.NDArray[np.float64],
        fs: float,
        window_s: float = 0.5,
        threshold: float = 10
):
    """
    Detects outlier noise in the input signal based on a threshold. The function evaluates the signal's deviation
    from the mean and identifies segments with excessive noise. It drops the last segment if ndarray shape
    is not a multiple of whole seconds.

    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.
        fs (float): Sampling frequency.
        window_s (float): Length of the detection window in seconds. Default is 0.5 seconds.
        threshold (float): Threshold for detecting outliers. Default is 10.

    Returns:
        np.ndarray: Boolean array indicating the presence of outlier noise for each detection window.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """

    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    x = x - np.nanmean(x, axis=1, keepdims=True)
    threshold_tukey = np.abs(np.nanpercentile(x, 90, axis=1) + \
         threshold * (np.nanpercentile(x, 90, axis=1) - np.nanpercentile(x, 10, axis=1)))

    b_idx = np.abs(x) > threshold_tukey[:, np.newaxis]

    detected_noise = np.array([
        buffer(b_ch, fs, segm_size=window_s, drop=True).sum(1) >= 1 for b_ch in b_idx
    ])

    if ndim == 1:
        detected_noise = detected_noise[0]

    return detected_noise

def detect_flat_line_segments(
        x: np.typing.NDArray[np.float64],
        fs: float,
        window_s:float = 0.5,
        threshold: float = 0.5e-6
):
    """
    Detects flat-line segments in the input signal. A flat-line segment is identified when the mean absolute
    difference of the signal within a detection window is below a specified threshold.  It
    drops the last segment if ndarray shape is not a multiple of whole seconds.

    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.
        fs (float): Sampling frequency.
        window_s (float): Length of the detection window in seconds. Default is 0.5 seconds.
        threshold (float): Threshold for detecting flat-line segments. Default is 0.5e-6.

    Returns:
        np.ndarray: Boolean array indicating the presence of flat-line segments for each detection window.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """
    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    xb = np.array([
        buffer(x_, fs, segm_size=window_s, drop=True) for x_ in x
    ])
    detected_flat_line = np.abs(np.diff(xb, axis=2)).mean(axis=2) < threshold

    if ndim == 1:
        detected_flat_line = detected_flat_line[0]

    return detected_flat_line


def detect_stim_segments(x: np.typing.NDArray[np.float64], fs: float, window_s:float = 1,
                         threshold_detection:float = 2000, freq_band: Tuple[float, float] = (80, 110,)):
    """
    Detects stimulation artifacts in the input signal. Calculates differential signal of the input signal.
    Spectral power of the differential signal between the bands provided in frequency band is
    thresholded  for each detection window.

    Parameters:
        x (np.ndarray): Input signal, either 1D or 2D array.
        fs (float): Sampling frequency.
        window_s (float): Length of the detection window in seconds. Default is 1 second.
        threshold_detection (float): Threshold for detecting stimulation artifacts. Default is 2000.
        freq_band (tuple): Frequency band to consider for artifact detection (low, high). Default is (80, 110).

    Returns:
        tuple:
            - np.ndarray: Boolean array indicating the presence of stimulation artifacts for each detection window.
            - np.ndarray: Spectral power within the specified frequency band for each detection window.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """
    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if x.ndim == 1:
        x = x[np.newaxis, :]

    x_diff = np.diff(x, axis=-1)     # difference signal highlights artificial pulses
    x_diff = np.concatenate(
        (x_diff, x_diff[:, -1].reshape(-1, 1)), axis=1,
    )

    xb =  np.array([
        buffer(x_, fs, segm_size=window_s, drop=True) for x_ in x_diff
    ])


    freq, psd = PSD(xb, fs=fs)
    psd_hf = psd[:, :, (freq > freq_band[0]) & (freq < freq_band[1])]
    psd_sum = np.sum(psd_hf, axis=-1)
    detected_stim = (psd_sum >= threshold_detection).astype(int)

    if ndim == 1:
        detected_stim = detected_stim[0]
        psd_sum = psd_sum[0]

    return detected_stim, psd_sum


def mask_segments_with_nans(x: np.typing.NDArray[np.float64], segment_mask: np.typing.NDArray[np.float64],
                            fs: float, window_s: float):
    """
    Masks EEG signal segments based on provided mask setting them to NaN.

    Parameters:
        x (np.ndarray): 1D or 2D array of EEG data with shape (n_channels, n_samples).
        fs (int): Sampling rate of the EEG signal in Hz.
        window_s (int): Duration of each masking segment in seconds.
        segment_mask (np.ndarray): Binary matrix of shape (n_channels, n_sec * window_s) where 1 indicates
                                   the presence of a stimulation artifact in given window.

    Returns:
        np.ndarray: EEG signal with artifact segments replaced by NaN.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """
    ndim = x.ndim
    if ndim == 0 or ndim > 2:
        raise ValueError("Input 'x' must be a 1D or nD numpy array.")

    if segment_mask.ndim != ndim:
        raise ValueError("Input 'merged_noise' must have same dimension as input signal 'x'.")

    if x.ndim == 1:
        x = x[np.newaxis, :]
        segment_mask = segment_mask[np.newaxis, :]


    n_channels, n_samples = x.shape
    samples_per_segment = int(np.round(fs * window_s))

    #  # Create index offsets for each segment
    window_len = segment_mask.shape[1]
    segment_indices = np.arange(window_len) * samples_per_segment
    segment_range = np.arange(samples_per_segment)

    # Find all artifact locations
    segment_offsets = segment_range[None, :] + segment_indices[:, None]     #shape: (n_seconds, samples_per_segment)
    channel_idx, second_idx = np.where(segment_mask == 1)
    sample_indices = segment_offsets[second_idx]  # shape: (num_artifacts, samples_per_segment)

    # Filter out segments that would exceed signal bounds
    valid_mask = sample_indices[:, -1] < n_samples
    channel_idx = channel_idx[valid_mask]
    sample_indices = sample_indices[valid_mask]

    # Apply NaNs to the artifact regions
    x_sub = x.copy()
    x_sub[channel_idx[:, None], sample_indices] = np.nan

    if ndim == 1:
        x_sub = x_sub[0]
    return x_sub


def detection_dilatation(mask: np.ndarray, extend_left: int = 2, extend_right: int = 2):
    """
    Extends True values in a boolean mask by a fixed number of positions to the left and right.

    Parameters:
        mask (np.ndarray): 1D or 2D boolean array indicating detection.
        extend_left (int): Number of positions to extend left of each detection.
        extend_right (int): Number of positions to extend right of each detection.

    Returns:
        np.ndarray: Extended boolean mask with same shape.

    Raises:
        ValueError: If the input signal is not 1D or 2D.
    """
    total_extend = extend_left + extend_right + 1
    structure = np.ones(total_extend, dtype=int)

    if mask.ndim == 1:
        return binary_dilation(mask, structure=structure, origin=0).astype(int)
    elif mask.ndim == 2:
        return np.array([
            binary_dilation(row, structure=structure, origin=0)
            for row in mask
        ]).astype(int)
    else:
        raise ValueError("Input 'mask' must be 1D or 2D.")
