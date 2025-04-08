import os
import sys
# Add the parent directory to Python path to find the rsHRF package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
from rsHRF import spm_dep, processing, parameters, basis_functions, utils

# Download ADHD dataset
print("Downloading ADHD dataset...")
adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_filename = adhd_dataset.func[0]

# Create output directory
output_dir = 'rsHRF_demo_output'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Set parameters
para = {}
para['TR'] = 2.0  # TR in seconds
para['T'] = 1     # No temporal interpolation
para['T0'] = 1    # No temporal shifting
para['order'] = 3 # AR model order
para['AR_lag'] = 1  # AR model lag
para['thr'] = 1   # Threshold
para['len'] = 24  # Length of HRF
para['min_onset_search'] = 4
para['max_onset_search'] = 8
para['localK'] = 2  # Local peak threshold
para['passband'] = [0.01, 0.08]  # Passband frequencies
para['passband_deconvolve'] = [0.01, 0.08]  # Passband frequencies for deconvolution
para['dt'] = para['TR'] / para['T']
para['lag'] = np.arange(np.fix(para['min_onset_search'] / para['dt']),
                      np.fix(para['max_onset_search'] / para['dt']) + 1,
                      dtype='int')
para['TD_DD'] = 1  # Add time and dispersion derivatives

print("Computing brain mask...")
# Compute and apply mask
mask_img = compute_epi_mask(func_filename)
bold_data = apply_mask(func_filename, mask_img)

print("Processing BOLD data...")
# Z-score the data
bold_sig = (bold_data - np.mean(bold_data, axis=0)) / np.std(bold_data, axis=0)

# Clean data: replace infs and NaNs with 0
bold_sig = np.nan_to_num(bold_sig, nan=0.0, posinf=0.0, neginf=0.0)

# Filter the data
bold_sig_filtered = processing.rest_filter.rest_IdealFilter(bold_sig, para['TR'], para['passband'])
bold_sig_deconv = processing.rest_filter.rest_IdealFilter(bold_sig, para['TR'], para['passband_deconvolve'])

# Clean filtered data
bold_sig_filtered = np.nan_to_num(bold_sig_filtered, nan=0.0, posinf=0.0, neginf=0.0)
bold_sig_deconv = np.nan_to_num(bold_sig_deconv, nan=0.0, posinf=0.0, neginf=0.0)

print("Estimating HRF...")
# Estimate HRF using canonical HRF with time derivative
para['estimation'] = 'canon2dd'
bf = basis_functions.basis_functions.get_basis_function(bold_sig_filtered.shape, para)
beta_hrf, event_bold = utils.hrf_estimation.compute_hrf(bold_sig_filtered, para, [], 2, bf=bf)
hrfa = np.dot(bf, beta_hrf[np.arange(0, bf.shape[1]), :])

# Get HRF parameters
nvar = hrfa.shape[1]
PARA = np.zeros((3, nvar))
for voxel_id in range(nvar):
    hrf1 = hrfa[:, voxel_id]
    PARA[:, voxel_id] = parameters.wgr_get_parameters(hrf1, para['TR'] / para['T'])

print("Deconvolving HRF...")
# Deconvolve HRF (using non-Wiener method)
data_deconv = np.zeros(bold_sig_deconv.shape)
nobs = bold_sig_deconv.shape[0]

for voxel_id in range(nvar):
    hrf = hrfa[:, voxel_id]
    H = np.fft.fft(np.append(hrf, np.zeros((nobs - max(hrf.shape), 1))), axis=0)
    M = np.fft.fft(bold_sig_deconv[:, voxel_id])
    data_deconv[:, voxel_id] = np.fft.ifft(H.conj() * M / (H * H.conj() + .1*np.mean((H * H.conj()))))

print("Plotting results...")
# Plot results for a sample voxel
voxel_to_plot = 1000  # Choose a sample voxel

plt.figure(figsize=(12, 8))

# Plot HRF
plt.subplot(2, 1, 1)
plt.plot(para['TR'] * np.arange(1, len(hrfa[:, voxel_to_plot]) + 1), hrfa[:, voxel_to_plot])
plt.title('Estimated HRF for Sample Voxel')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot BOLD and deconvolved signal
plt.subplot(2, 1, 2)
plt.plot(para['TR'] * np.arange(1, nobs + 1), bold_sig[:, voxel_to_plot], label='BOLD')
plt.plot(para['TR'] * np.arange(1, nobs + 1), data_deconv[:, voxel_to_plot], label='Deconvolved', alpha=0.8)
if len(event_bold) > 0 and voxel_to_plot < len(event_bold):
    events = np.zeros(nobs)
    event_indices = np.array(event_bold[voxel_to_plot]).astype(int)
    events[event_indices] = 1
    plt.stem(para['TR'] * np.arange(1, nobs + 1), events, linefmt='k-', markerfmt='kd', basefmt='k-', label='Events')
plt.title('BOLD and Deconvolved Signal with Events')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rsHRF_results.png'))
plt.close()

print('Analysis complete Results saved in', output_dir)