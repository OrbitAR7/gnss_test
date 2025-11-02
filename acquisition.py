import numpy as np
from scipy.fft import fft, ifft

def gnss_acquire(data, fif, fs, prn, ta, f_search=None):
    """
    Acquire GNSS signal.
    
    Returns: (doppler_hz, code_offset_sec, cn0_db_hz)
    """
    code_period = 1e-3
    n_code = int(fs * code_period)
    n_acq = int(fs * ta)
    n_codes = n_acq // n_code
    n_fft = 2**int(np.ceil(np.log2(n_acq)))
    
    if f_search is None:
        f_search = np.arange(-7000, 7001, 1/(4*ta))
    
    data_acq = data[:n_fft]
    t_vec = np.arange(n_fft) / fs
    
    # Oversample PRN code
    prn_full = np.tile(prn, n_codes)
    prn_os = oversample_code(prn_full, fs, n_acq, len(prn_full))
    prn_fft = np.conj(fft(prn_os, n_fft))
    
    # Search over Doppler frequencies
    corr_map = np.zeros((n_code, len(f_search)))
    
    for i, fd in enumerate(f_search):
        local = np.exp(-1j * 2 * np.pi * (fif + fd) * t_vec)
        x_bb = data_acq * local
        x_fft = fft(x_bb, n_fft)
        corr = ifft(x_fft * prn_fft)
        corr_map[:, i] = np.abs(corr[:n_code])**2
    
    # Find peak
    peak = np.max(corr_map)
    idx_code, idx_freq = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    
    doppler = f_search[idx_freq]
    code_offset = idx_code / fs
    
    # Estimate C/N0
    cn0 = estimate_cn0(corr_map, idx_code, idx_freq, peak, ta, fs)
    
    return doppler, code_offset, cn0

def oversample_code(code, fs, n_samples, n_chips):
    """Oversample spreading code to match sampling rate."""
    chip_rate = 1.023e6
    samples_per_chip = fs / chip_rate
    indices = np.arange(n_samples) / samples_per_chip
    code_indices = np.floor(indices).astype(int) % len(code)
    return code[code_indices]

def estimate_cn0(corr_map, idx_code, idx_freq, peak, ta, fs):
    """Estimate carrier-to-noise ratio."""
    chip_interval = 1e-3 / 1023
    n_code = corr_map.shape[0]
    
    # Exclude region around peak
    idx_f_width = max(1, int(2 / (ta * (corr_map.shape[1] / 1000))))
    idx_t_width = int(np.ceil(chip_interval * fs))
    
    mask = np.ones_like(corr_map, dtype=bool)
    mask[max(0, idx_code-idx_t_width):min(n_code, idx_code+idx_t_width+1), idx_freq] = False
    mask[idx_code, max(0, idx_freq-idx_f_width):min(corr_map.shape[1], idx_freq+idx_f_width+1)] = False
    
    noise_power = np.mean(corr_map[mask])
    signal_power = peak - noise_power
    cn0 = 10 * np.log10(signal_power / (noise_power * ta))
    
    return cn0
