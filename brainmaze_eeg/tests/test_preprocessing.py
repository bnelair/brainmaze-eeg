
import pytest

import numpy as np
import scipy.signal as signal


from brainmaze_eeg.preprocessing import (
    replace_nans_with_median,
    channel_data_rate_thresholding,
    filter_powerline,
    detect_outlier_segments,
    detect_powerline_segments,
    detect_flat_line_segments,
    detect_stim_segments,
    mask_segments_with_nans,
    detection_dilatation
)

def test_replace_nans_with_medians():
    # Test 2D input with drop rate below threshold
    x_2d = np.array([[np.nan, 2.0, 2.0, 2.0, np.nan], [4.0, 5.0, 6.0, np.nan, np.nan]])

    result_2d, mask = replace_nans_with_median(x_2d)
    assert (np.isnan(result_2d).sum(1) == [0, 0]).all(), "2D input below threshold should not be masked"
    assert result_2d[0, 0] == result_2d[0, -1] == 2.0, "First element should be 2.0"
    assert result_2d[1, -1] == result_2d[1, -2] == 5.0, "First element should be 4.0"

    x_1d_nans = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    result_1d, mask = replace_nans_with_median(x_1d_nans)
    assert np.isnan(result_1d).sum() == 5, "1D input with all NaNs should be fully masked"

    x_1d_num = np.random.randn(10)
    result_1d, mask = replace_nans_with_median(x_1d_num)
    assert np.isnan(result_1d).sum() == 0, "1D input with all NaNs should be fully masked"
    assert (result_1d == x_1d_num).all()

def test_channel_data_rate_thresholding():
    # Test 1D input with drop rate below threshold
    x_1d = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    dr_threshold = 0.5
    result = channel_data_rate_thresholding(x_1d, dr_threshold)
    assert np.isnan(result).sum() == np.isnan(x_1d).sum(), "1D input below threshold should not be masked"

    # Test 1D input with drop rate above threshold
    x_1d_high_drop = np.array([np.nan, np.nan, 3.0, np.nan, np.nan])
    dr_threshold = 0.5
    result = channel_data_rate_thresholding(x_1d_high_drop, dr_threshold)
    assert np.all(np.isnan(result)), "1D input above threshold should be fully masked"

    # Test 2D input with drop rate below threshold
    x_2d = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    dr_threshold = 0.5
    result = channel_data_rate_thresholding(x_2d, dr_threshold)
    assert np.isnan(result).sum() == np.isnan(x_2d).sum(), "2D input below threshold should not be masked"

    # Test 2D input with drop rate above threshold
    x_2d_high_drop = np.array([[np.nan, np.nan, 3.0], [np.nan, np.nan, np.nan]])
    dr_threshold = 0.5
    result = channel_data_rate_thresholding(x_2d_high_drop, dr_threshold)
    assert np.all(np.isnan(result[0, :])), "2D input row above threshold should be fully masked"
    assert np.all(np.isnan(result[1, :])), "2D input row above threshold should be fully masked"

    # Test 2D input with mixed drop rates
    x_2d_mixed = np.array([[1.0, 2.0, np.nan], [np.nan, np.nan, np.nan]])
    dr_threshold = 0.5
    result = channel_data_rate_thresholding(x_2d_mixed, dr_threshold)
    assert np.isnan(result[0, :]).sum() == 1, "First row should have one NaN"
    assert np.all(np.isnan(result[1, :])), "Second row should be fully masked"
    assert np.isnan(result).sum() == 4, "Total NaNs should be 4"


def test_filter_powerline():
    # Parameters
    fs = 1000  # Sampling frequency in Hz
    duration = 2  # Duration of the signal in seconds
    frequency_powerline = 60  # Powerline frequency in Hz

    # Generate a signal with 60 Hz sinusoidal component
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal_60hz = np.sin(2 * np.pi * frequency_powerline * t)

    # Add some noise to the signal
    noise = np.random.normal(0, 0.1, signal_60hz.shape)

    signal_with_noise = signal_60hz + noise

    signal_with_nan = signal_with_noise.copy()
    signal_with_nan[100:1000] = np.nan

    # Apply the notch filter
    filtered_signal = filter_powerline(signal_with_noise, fs, frequency_powerline)
    filtered_signal_with_nan = filter_powerline(signal_with_nan, fs, frequency_powerline)

    assert np.isnan(filtered_signal_with_nan).sum() == np.isnan(signal_with_nan).sum(), "Filtered signal with NaNs should have the same number of NaNs"
    assert np.nansum(np.abs(filtered_signal_with_nan)) < np.nansum(np.abs(signal_with_nan)), "Filtered signal with NaNs should have smaller power than original with 60 Hz"


    # Compute the power spectrum before filtering
    freqs, psd_before = signal.welch(signal_with_noise, fs, nperseg=fs)
    freqs, psd_before_nan = signal.welch(signal_with_nan, fs, nperseg=fs)

    # Compute the power spectrum after filtering
    freqs, psd_after = signal.welch(filtered_signal, fs, nperseg=fs)

    # Find the power at 60 Hz before and after filtering
    idx_60hz = np.argmin(np.abs(freqs - frequency_powerline))
    power_before = psd_before[idx_60hz]
    power_after = psd_after[idx_60hz]

    # Assert that the power at 60 Hz is reduced after filtering
    assert power_after < power_before, f"Power at 60 Hz was not reduced: before={power_before}, after={power_after}"

    print(f"Test passed: Power at 60 Hz reduced from {power_before:.2e} to {power_after:.2e}")


def test_detect_powerline_segments():
    # Parameters
    fs = 250  # Sampling frequency in Hz
    duration = 100  # Duration of the signal in seconds
    frequency_powerline = 60  # Powerline frequency in Hz

    # Generate a signal with 60 Hz sinusoidal component
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal_60hz = np.sin(2 * np.pi * frequency_powerline * t)

    x = 0.1*np.random.randn(signal_60hz.shape[0])
    x[:int(x.shape[0]/2)] += signal_60hz[:int(x.shape[0]/2)]

    y = detect_powerline_segments(x, fs, window_s=1.0, powerline_freq=60)
    assert y.sum() == y.shape[0] / 2

    x_merged = np.stack([x, x[::-1]], 0)
    y_merged = detect_powerline_segments(x_merged, fs, window_s=1.0, powerline_freq=60)
    assert np.all(y_merged[0] == y)
    assert np.all(y_merged[1] == y[::-1])


def test_detect_outlier_segments():
    np.random.seed(42)  # Fixed seed for reproducibility
    fs = 250  # Sampling frequency in Hz
    duration = 60  # Duration of the signal in seconds
    idx = np.arange(fs+10, fs + 20)

    x = 0.001 * np.random.randn(int(fs * duration))  # Very small noise level
    b, a = signal.butter(4, 50, btype='low', fs=fs)
    x = signal.filtfilt(b, a, x)

    x[idx] = 0.1  # Much smaller outlier value that won't skew global statistics

    x = np.stack([x, x[::-1]], 0)

    y = detect_outlier_segments(x, fs, window_s=1)

    assert y.shape == (2, duration), "Output shape should be (2, duration)"
    assert y[0, 0] == False, "1st second segment should not be detected as noise"
    assert y[0, 1] == True, "2nd second segment should be detected as noise"
    assert y[0, 2] == False, "3rd second segment should not be detected as noise"
    assert np.all(y[0] == y[1][::-1]), "Output should be the same for both channels"


def test_detect_flat_line_segments():
    fs = 250
    duration = 60

    x = 1*np.random.randn(int(fs * duration))
    # a genuinely flat segment: sample-to-sample changes well below the 0.5 uV threshold.
    # (1e-6 scaling is ~1 uV of noise, whose mean |diff| ~1.1 uV is *above* threshold and
    # so is not flat under the documented definition.)
    x[fs:2*fs] *= 1e-9

    x = np.stack([x, x[::-1]], 0)

    y = detect_flat_line_segments(x, fs, window_s=1)

    assert y.shape == (2, duration,), "Output shape should be (duration,)"
    assert y[0, 0] == False, "1st second segment should not be detected as flat line"
    assert y[0, 1] == True, "2nd second segment should be detected as flat line"
    assert y[0, 2] == False, "3rd second segment should not be detected as flat line"
    assert np.all(y[0] == y[1][::-1]), "Output should be the same for both channels"


def test_detect_flat_line_segments_oscillation_is_not_flat():
    # regression: an oscillation returns to near its starting value each cycle, so the
    # mean *of* the differences telescopes to ~0. The detector must use the mean of the
    # *absolute* differences, otherwise a large-amplitude periodic signal reads as flat.
    fs = 250
    t = np.arange(0, 10, 1 / fs)
    x = 100e-6 * np.sin(2 * np.pi * 10 * t)   # 100 uV, 10 Hz -- clearly not flat

    y = detect_flat_line_segments(x, fs, window_s=0.5)
    assert not y.any(), "an oscillating signal must never be flagged as flat line"

    # and a true DC segment of the same duration still is flat
    assert detect_flat_line_segments(np.zeros_like(x), fs, window_s=0.5).all()

def test_detect_stim_segments():
    # Parameters
    fs = 250  # Hz
    duration = 60  # seconds
    stim_freq = 145 # Hz
    detection_window = 1
    stim_start_sec, stim_end_sec = 5, 15

    # Generate EEG + add stim artifact
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal_145hz = 100 * np.sin(2 * np.pi * stim_freq * t)

    x = 0.1*np.random.randn(signal_145hz.shape[0])
    x[int(stim_start_sec*fs):int(stim_end_sec*fs)] += signal_145hz[int(stim_start_sec*fs):int(stim_end_sec*fs)]

    y, psd_sum = detect_stim_segments(x, fs, window_s=detection_window)

    assert y.shape[0] == x.shape[0]/fs/detection_window, ""
    assert np.all(y[int(stim_start_sec/detection_window):int(stim_end_sec/detection_window)] == 1)

    x2d = np.stack([x, x[::-1]], 0)
    y2, psd_sum2 = detect_stim_segments(x2d, fs, window_s=detection_window)

    assert np.all(y2[0] == y)
    assert np.all(y2[1, int((duration-stim_end_sec)/detection_window):int((duration-stim_start_sec)/detection_window)] == 1)


def test_mask_segments_with_nans():
    fs = 500
    n_sec = 60
    n_samples = fs * n_sec
    n_channels = 2
    segment_length = 1

    x2d = np.random.randn(n_channels, n_samples)

    merged_noise = np.zeros((n_channels, n_sec))
    merged_noise[0, [1, 3]] = 1
    merged_noise[1, [2]] = 1

    x_clean2d = mask_segments_with_nans(x2d, merged_noise, fs, segment_length)

    assert np.isnan(x_clean2d[0, 1 * fs:2 * fs]).all()
    assert np.isnan(x_clean2d[0, 3 * fs:4 * fs]).all()
    assert np.isnan(x_clean2d[1, 2 * fs:3 * fs]).all()

    assert ~(np.isnan(x_clean2d[0, 0:fs]).any())
    assert ~(np.isnan(x_clean2d[1, 0:fs]).any())

    x = np.random.randn(n_samples)
    merged_noise = np.zeros(n_sec)
    merged_noise[20] = 1  # noise in second 1
    x_clean = mask_segments_with_nans(x, merged_noise, fs, segment_length)

    assert (~np.isnan(x_clean[fs:20 * fs])).all()
    assert np.isnan(x_clean[20 * fs: 21*fs]).all()
    assert (~np.isnan(x_clean[21*fs:])).all()


def test_detection_dilatation():
    nsec = 60
    det_vector = np.zeros(nsec)
    det_vector[10] = 1
    det_vector[-1] = 1

    det_vector_extend = detection_dilatation(det_vector, extend_left=2, extend_right=2)
    assert np.sum(det_vector_extend) == 8
    assert np.all(det_vector_extend[8:13] == 1)
    assert np.all(det_vector_extend[57:59] == 1)
    assert np.all(det_vector_extend[[7,13,56]] == 0)

    det_matrix = np.stack([det_vector, det_vector[::-1]], 0)
    det_matrix_extend = detection_dilatation(det_matrix, extend_left=2, extend_right=2)
    assert np.all(np.sum(det_matrix_extend, axis=1) == 8)

