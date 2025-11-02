import numpy as np
from acquisition import oversample_code

def track_signal(data, t_vec, prn, fif, fs, fl1, ta, fd_init, ts_init, cn0_init, 
                 pll_bn=10.0, dll_bn=0.1):
    """Track GNSS signal with PLL and DLL."""
    n_total = len(data)
    n_acq = int(fs * ta)
    teml = 0.5 * (1e-3 / 1023)
    
    # Initialize
    results = {
        'i_prompt': [], 'q_prompt': [],
        'doppler': [], 'code_phase': [],
        'cn0': []
    }
    
    # Configure PLL (3rd order)
    pll_state = configure_loop_filter(pll_bn, ta, order=3)
    pll_state['x'] = np.array([2*np.pi*fd_init, 2*np.pi*fd_init])
    
    # DLL parameters
    sigma_iq = 10**(cn0_init/20) / np.sqrt(2 * ta)
    
    ts = ts_init
    theta = 0.0
    fd = fd_init
    
    idx = int(ts * fs)
    
    while idx + n_acq < n_total:
        # Adjust integration time for Doppler
        ta_adj = ta / (1 + fd / fl1)
        n_acq_adj = int(fs * ta_adj)
        
        if idx + n_acq_adj >= n_total:
            break
        
        # Get data segment
        data_seg = data[idx:idx + n_acq_adj]
        t0 = t_vec[idx]
        
        # Correlate
        sp, se, sl = correlate(data_seg, t0, fif, fs, ta_adj, ts, fd, theta, teml, prn)
        
        results['i_prompt'].append(sp.real)
        results['q_prompt'].append(sp.imag)
        
        # Update PLL
        pll_state['ip'] = sp.real
        pll_state['qp'] = sp.imag
        pll_state, vk = update_pll(pll_state)
        fd_new = vk / (2 * np.pi)
        
        # Update DLL
        dll_error = compute_dll_error(sp, se, sl, sigma_iq, teml * (1 - fd/fl1))
        vp = vk / (2 * np.pi * fl1)
        v_code = 4 * dll_bn * dll_error + vp
        
        # Store results
        results['doppler'].append(fd_new)
        results['code_phase'].append(ts)
        
        # Estimate C/N0
        cn0 = 10 * np.log10(np.abs(sp)**2 / (2 * sigma_iq**2 * ta))
        results['cn0'].append(cn0)
        
        # Update for next iteration
        dt = n_acq_adj / fs
        theta = theta + dt * vk
        ts = ts + (1 - v_code) * 1e-3
        fd = fd_new
        
        idx += n_acq_adj
    
    # Convert lists to arrays
    for key in results:
        results[key] = np.array(results[key])
    
    return results

def correlate(data, t0, fif, fs, ta, ts, fd, theta, teml, prn):
    """Compute early, prompt, late correlations."""
    n = len(data)
    t = t0 + np.arange(n) / fs
    
    # Local oscillator
    local = np.exp(-1j * (2*np.pi*(fif*t + fd*(t - t0)) + theta))
    
    # Generate codes
    n_chips = len(prn)
    chip_interval = 1e-3 / n_chips
    n_codes = int(np.ceil(ta / 1e-3))
    prn_full = np.tile(prn, n_codes)
    
    code_p = oversample_code_tracking(prn_full, fs, n, (t0 - ts) / chip_interval)
    code_e = oversample_code_tracking(prn_full, fs, n, (t0 - (ts - teml)) / chip_interval)
    code_l = oversample_code_tracking(prn_full, fs, n, (t0 - (ts + teml)) / chip_interval)
    
    # Correlate
    sp = np.sum(data * local * code_p)
    se = np.sum(data * local * code_e)
    sl = np.sum(data * local * code_l)
    
    return sp, se, sl

def oversample_code_tracking(code, fs, n_samples, offset_chips):
    """Oversample code with offset."""
    chip_rate = 1.023e6
    samples_per_chip = fs / chip_rate
    indices = (np.arange(n_samples) + offset_chips * samples_per_chip)
    indices = indices / samples_per_chip
    code_indices = np.floor(indices).astype(int) % len(code)
    return code[code_indices]

def configure_loop_filter(bn, ta, order=3):
    """Configure loop filter state-space model."""
    state = {'ta': ta, 'order': order}
    
    if order == 3:
        a = 1.2 * bn
        b = a**2 / 2
        k = 2 * a
        
        # Simplified 3rd order filter
        state['ad'] = np.array([[1, ta], [0, 1]])
        state['bd'] = np.array([k*ta + k*ta**2/2, k*ta])
        state['cd'] = np.array([1, 0])
        state['dd'] = k
    
    return state

def update_pll(state):
    """Update PLL with arctangent discriminator."""
    error = np.arctan2(state['qp'], state['ip'])
    x_new = state['ad'] @ state['x'] + state['bd'] * error
    vk = state['cd'] @ state['x'] + state['dd'] * error
    state['x'] = x_new
    return state, vk

def compute_dll_error(sp, se, sl, sigma_iq, tc):
    """Compute DLL error with early-late discriminator."""
    power = np.abs(sp)**2
    early_corr = se.real * sp.real + se.imag * sp.imag
    late_corr = sl.real * sp.real + sl.imag * sp.imag
    
    c = (tc / 2) / (power - 2 * sigma_iq**2)
    error = c * (early_corr - late_corr)
    
    return error
