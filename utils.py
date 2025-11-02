import numpy as np

def load_if_data(filename, fs, duration=60):
    """Load IF data from binary file."""
    n_samples = int(fs * duration)
    n_samples = (n_samples // 16) * 16  # Round to multiple of 16
    
    try:
        with open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=np.int16, count=n_samples)
        
        if len(data) != n_samples:
            raise ValueError(f"Expected {n_samples} samples, got {len(data)}")
        
        t_vec = np.arange(len(data)) / fs
        return data.astype(float), t_vec
    
    except FileNotFoundError:
        print(f"File {filename} not found. Generating synthetic data...")
        return generate_synthetic_data(fs, duration)

def generate_synthetic_data(fs, duration):
    """Generate synthetic GPS signal for testing."""
    n_samples = int(fs * duration)
    n_samples = (n_samples // 16) * 16
    t_vec = np.arange(n_samples) / fs
    
    # Simulate weak GPS signal with noise
    noise = np.random.randn(n_samples) * 100
    
    # Add weak carrier (simulating GPS signal)
    fif = 1610476.19
    signal = 10 * np.cos(2 * np.pi * fif * t_vec)
    
    data = signal + noise
    
    print("Note: Using synthetic data for demonstration")
    return data, t_vec

def compute_cn0(i_prompt, q_prompt, ta):
    """Compute C/N0 from I/Q samples."""
    power = i_prompt**2 + q_prompt**2
    mean_power = np.mean(power)
    std_power = np.std(power)
    cn0 = 10 * np.log10(mean_power / (std_power * ta))
    return cn0

def if2iq(x, fs, fif):
    """Convert IF samples to baseband I/Q."""
    n = len(x)
    t = np.arange(n) / fs
    
    i = x * np.sqrt(2) * np.cos(2 * np.pi * fif * t)
    q = -x * np.sqrt(2) * np.sin(2 * np.pi * fif * t)
    
    # Decimate by 2
    i = i[::2]
    q = q[::2]
    
    return i, q

def iq2if(i, q, fs_baseband, fif):
    """Convert baseband I/Q to IF samples."""
    n = len(i)
    fs = 2 * fs_baseband
    t = np.arange(2 * n) / fs
    
    # Interpolate I and Q
    i_interp = np.repeat(i, 2)
    q_interp = np.repeat(q, 2)
    
    x = i_interp * np.sqrt(2) * np.cos(2 * np.pi * fif * t) - \
        q_interp * np.sqrt(2) * np.sin(2 * np.pi * fif * t)
    
    return x
