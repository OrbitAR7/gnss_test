import numpy as np

def generate_lfsr(n, taps, initial_state):
    """Generate LFSR sequence."""
    m = 2**n - 1
    seq = np.zeros(m, dtype=int)
    state = initial_state.copy()
    
    feedback = np.zeros(n, dtype=int)
    feedback[taps] = 1
    
    for i in range(m):
        seq[i] = state[-1]
        new_bit = np.sum(feedback * state) % 2
        state = np.roll(state, 1)
        state[0] = new_bit
    
    return seq

def generate_prn(tx_id):
    """Generate GPS L1 C/A PRN code for given satellite."""
    taps1 = np.array([2, 9])  # G1: 1 + D^3 + D^10
    taps2 = np.array([1, 2, 5, 7, 8, 9])  # G2: 1 + D^2 + D^3 + D^6 + D^8 + D^9 + D^10
    initial = np.ones(10, dtype=int)
    
    g1 = generate_lfsr(10, taps1, initial)
    g2 = generate_lfsr(10, taps2, initial)
    
    delays = [5, 6, 7, 8, 17, 18, 139, 140, 141, 251, 252, 254, 255,
              256, 257, 258, 469, 470, 471, 472, 473, 474, 509, 512,
              513, 514, 515, 516, 859, 860, 861, 862, 863]
    
    g2_shifted = np.roll(g2, delays[tx_id - 1])
    prn = 2 * ((g1 + g2_shifted) % 2) - 1
    
    return prn
