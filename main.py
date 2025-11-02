import numpy as np
from acquisition import gnss_acquire
from tracking import track_signal
from signals import generate_prn
from utils import load_if_data

def main():
    # Configuration
    fs = 40e6 / 7  # Sampling frequency (Hz)
    fif = 1610476.19  # Intermediate frequency (Hz)
    fl1 = 154 * 10.23e6  # L1 carrier frequency (Hz)
    ta = 1e-3  # Integration time (s)
    
    # Load data
    print("Loading IF data...")
    data, t_vec = load_if_data('SiGe_Bands-L1.dat', fs, duration=60)
    
    # Satellites to track
    tx_ids = [1, 7, 8, 11, 28, 30]
    
    results = {}
    
    for tx_id in tx_ids:
        print(f"\n=== Satellite {tx_id} ===")
        
        # Generate PRN code
        prn = generate_prn(tx_id)
        
        # Coarse acquisition
        print("Coarse acquisition...")
        fd_coarse, ts_coarse, cn0 = gnss_acquire(
            data, fif, fs, prn, ta, 
            f_search=np.arange(-40e3, -10e3, 1/(4*ta))
        )
        print(f"  Doppler: {fd_coarse:.1f} Hz, C/N0: {cn0:.1f} dB-Hz")
        
        # Fine acquisition
        print("Fine acquisition...")
        fd, ts, _ = gnss_acquire(
            data, fif, fs, prn, 10e-3,
            f_search=fd_coarse + np.arange(-250, 252, 2)
        )
        print(f"  Refined Doppler: {fd:.1f} Hz")
        
        # Track signal
        print("Tracking...")
        tracking_results = track_signal(
            data, t_vec, prn, fif, fs, fl1, ta,
            fd_init=fd, ts_init=ts, cn0_init=cn0,
            pll_bn=10.0, dll_bn=0.1
        )
        
        results[tx_id] = tracking_results
        
        # Print summary
        mean_cn0 = np.mean(tracking_results['cn0'])
        print(f"  Mean C/N0: {mean_cn0:.1f} dB-Hz")
        print(f"  Tracked for {len(tracking_results['i_prompt'])*ta:.1f} seconds")
    
    print("\n=== Tracking Complete ===")
    return results

if __name__ == '__main__':
    results = main()
