"""Channel transformations - verbatim from the original ``dataload.py``.

These functions define how the raw complex channel matrix is turned into the
model inputs (amplitude/phase, real/imaginary, delay-angle domain).  They are
kept **exactly** as in the official LWLM repository.
"""

import numpy as np


def channel_normalization(H):
    P_current = np.sum(np.abs(H) ** 2)
    # alpha = np.sqrt((H.shape[1] * H.shape[2]) / P_current)
    alpha = np.sqrt((H.shape[2]) / P_current)

    # Scale the matrix so that the total power becomes N*M
    H_normalized = H * alpha
    return H_normalized


def channel_normalization_with_noise(H, snr_db):
    P_current = np.sum(np.abs(H) ** 2)
    # alpha = np.sqrt((H.shape[1] * H.shape[2]) / P_current)
    # alpha = np.sqrt((H.shape[2]) / P_current)
    alpha = np.sqrt((H.shape[1] * H.shape[2] / 2) / P_current)

    # Scale the matrix so that the total power becomes N*M
    H_normalized = H * alpha

    # Signal power (normalized to approximately H.shape[2])
    signal_power = np.mean(np.abs(H_normalized) ** 2)

    # Compute noise power
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate complex Gaussian white noise (real and imaginary parts are
    # independent, mean 0, variance noise_power/2)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))

    # Add noise
    H_noisy = H_normalized + noise

    return H_noisy


def freq_to_delay_angle_domain(H_freq):
    H_complex = H_freq[0] + 1j * H_freq[1]

    H_delay = np.fft.ifft(H_complex, axis=0)

    H_delay_angle_complex = np.fft.fft(H_delay, axis=1) / np.sqrt(H_delay.shape[0])  # / 2

    return H_delay_angle_complex
