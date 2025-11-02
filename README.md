# GNSS Signal Processing Toolkit

Python implementation of GPS L1 C/A signal acquisition and tracking.

## Installation

```bash
pip install numpy scipy
```

## Usage

1. Download GPS data from https://sdr.ion.org/api-sample-data.html
2. Place data file in project folder
3. Update `main.py` line 11 with your filename
4. Run:

```bash
python main.py
```

## What It Does

- **Acquires** GPS satellites (finds Doppler frequency and code phase)
- **Tracks** carrier and code using PLL/DLL loops
- **Outputs** C/N0, Doppler, and tracking results

## Core Functions

- `signals.py` - GPS PRN code generation
- `acquisition.py` - FFT-based signal acquisition
- `tracking.py` - PLL/DLL tracking loops
- `utils.py` - Data loading helpers

## Example Results

Tested with `mysteryData3.bin` (60 seconds, 5.7 MHz sampling):

```
=== Satellite 7 ===
Coarse acquisition...
  Doppler: -12000.0 Hz, C/N0: 41.1 dB-Hz
Fine acquisition...
  Refined Doppler: -12028.0 Hz
Tracking...
  Mean C/N0: 46.9 dB-Hz
  Tracked for 60.0 seconds

=== Satellite 28 ===
Coarse acquisition...
  Doppler: -13500.0 Hz, C/N0: 40.8 dB-Hz
Fine acquisition...
  Refined Doppler: -13440.0 Hz
Tracking...
  Mean C/N0: 47.1 dB-Hz
  Tracked for 60.0 seconds
```

Successfully tracked 6 satellites: PRN 1, 7, 8, 11, 28, 30

## Get GPS Data

**Free sources:**
- ION GNSS SDR: https://sdr.ion.org/api-sample-data.html
- GNSS-SDR: https://sourceforge.net/projects/gnss-sdr/files/data/

Recommended files:
- `SiGe_Bands-L1.dat` 
- `HackRF_Bands-L1.int8` 

**Note:** Data files not included in repo (too large for GitHub)

## References

- IS-GPS-200: GPS Interface Specification
- Kaplan & Hegarty: Understanding GPS/GNSS
- Borre et al.: Software-Defined GPS Receiver
