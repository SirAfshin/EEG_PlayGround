from typing import Dict, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from torcheeg.transforms.base_transform import EEGTransform
import io 
import random
import scipy.signal as signal
# import torchaudio ## TODO: install
# import pywt ## TODO: install # PyWavelets for Continuous Wavelet Transform

# Spectrogram Transforms
class STFTSpectrogram(EEGTransform):
    r'''
    A transform method to convert EEG signals of each channel into spectrograms using Short-Time Fourier Transform (STFT).

    Args:
        n_fft (int): The size of the FFT window (default: 64).
        hop_length (int): The hop length between successive windows (default: 32).
        window (str): The type of window to apply (default: 'hann').
        contourf (bool): Whether to output the spectrogram as an image with filled contours (default: False).

    Returns:
        A dictionary containing the transformed EEG data as spectrograms.
    '''
    def __init__(self,
                 n_fft: int = 64,
                 hop_length: int = 32,
                 window: str = 'hann',
                 contourf: bool = False,
                 apply_to_baseline: bool = False):
        super(STFTSpectrogram, self).__init__(apply_to_baseline=apply_to_baseline)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.contourf = contourf

        # Create the window function based on the input
        self.window_fn = torch.hann_window(n_fft) if window == 'hann' else torch.ones(n_fft)

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        channel_list = []
        for channel in eeg:
            channel_list.append(self.opt(channel))
        return np.array(channel_list)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        # Convert the EEG signal into a tensor
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)

        # Compute the STFT for the EEG signal
        stft_result = torch.stft(eeg_tensor, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window_fn, return_complex=True)

        # Calculate the magnitude (absolute value) of the STFT
        spectrogram = stft_result.abs()

        # Optionally, apply logarithmic scaling
        log_spectrogram = torch.log(spectrogram + 1e-7)

        if self.contourf:
            # Visualization option: Generate filled contour plots for each channel
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(log_spectrogram.numpy(), aspect='auto', origin='lower', cmap='jet')
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
            ax.set_title('Spectrogram')

            # Save the plot as an image in a byte buffer
            with io.BytesIO() as buf:
                fig.savefig(buf, format='raw')
                buf.seek(0)
                img_data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                img_w, img_h = fig.canvas.get_width_height()
                img_data = img_data.reshape((int(img_h), int(img_w), -1))

            return img_data

        return log_spectrogram.numpy()

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'window': self.window,
                'contourf': self.contourf
            })

class MelSpectrogram(EEGTransform):
    '''
    A transform method to convert EEG signals of each channel into Mel spectrograms.

    Args:
        sample_rate (int): The sampling rate of the EEG signal (default: 1000).
        n_fft (int): The size of the FFT window (default: 64).
        hop_length (int): The hop length between successive windows (default: 32).
        n_mels (int): The number of Mel frequency bins (default: 40).
        contourf (bool): Whether to output the spectrogram as an image with filled contours (default: False).

    Returns:
        A dictionary containing the transformed EEG data as Mel spectrograms.
    '''
    def __init__(self,
                 sample_rate: int = 1000,
                 n_fft: int = 64,
                 hop_length: int = 32,
                 n_mels: int = 40,
                 contourf: bool = False,
                 apply_to_baseline: bool = False):
        super(MelSpectrogram, self).__init__(apply_to_baseline=apply_to_baseline)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.contourf = contourf

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        channel_list = []
        for channel in eeg:
            channel_list.append(self.opt(channel))
        return np.array(channel_list)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        # Convert the EEG signal into a tensor
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)

        # Use torchaudio to compute Mel Spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )(eeg_tensor)

        # Optionally, apply logarithmic scaling
        log_mel_spec = torch.log(mel_spec + 1e-7)

        if self.contourf:
            # Visualization option: Generate filled contour plots for each channel
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(log_mel_spec.numpy(), aspect='auto', origin='lower', cmap='jet')
            ax.set_xlabel('Time')
            ax.set_ylabel('Mel Frequency')
            ax.set_title('Mel Spectrogram')

            # Save the plot as an image in a byte buffer
            with io.BytesIO() as buf:
                fig.savefig(buf, format='raw')
                buf.seek(0)
                img_data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                img_w, img_h = fig.canvas.get_width_height()
                img_data = img_data.reshape((int(img_h), int(img_w), -1))

            return img_data

        return log_mel_spec.numpy()

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'sample_rate': self.sample_rate,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels,
                'contourf': self.contourf
            })

class CWTSpectrogram(EEGTransform):
    '''
    A transform method to convert EEG signals into spectrograms using Continuous Wavelet Transform (CWT).

    Args:
        wavelet (str): The type of wavelet to use (default: 'cmor').
        scales (list or np.ndarray): The scales for the wavelet transform (default: 1 to 128).
        contourf (bool): Whether to output the spectrogram as an image with filled contours (default: False).

    Returns:
        A dictionary containing the transformed EEG data as CWT spectrograms.
    '''
    def __init__(self,
                 wavelet: str = 'cmor',
                 scales: np.ndarray = np.arange(1, 128),
                 contourf: bool = False,
                 apply_to_baseline: bool = False):
        super(CWTSpectrogram, self).__init__(apply_to_baseline=apply_to_baseline)
        self.wavelet = wavelet
        self.scales = scales
        self.contourf = contourf

    def __call__(self,
                 *args,
                 eeg: np.ndarray,
                 baseline: Union[np.ndarray, None] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        channel_list = []
        for channel in eeg:
            channel_list.append(self.opt(channel))
        return np.array(channel_list)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        # Perform CWT for each channel
        cwt_result = []
        for signal_channel in eeg:
            coeffs, _ = pywt.cwt(signal_channel, self.scales, self.wavelet)
            cwt_result.append(coeffs)
        
        cwt_result = np.array(cwt_result)

        if self.contourf:
            # Visualization option: Generate filled contour plots for each channel
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(cwt_result, aspect='auto', origin='lower', cmap='jet')
            ax.set_xlabel('Time')
            ax.set_ylabel('Scale')
            ax.set_title('CWT Spectrogram')

            # Save the plot as an image in a byte buffer
            with io.BytesIO() as buf:
                fig.savefig(buf, format='raw')
                buf.seek(0)
                img_data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                img_w, img_h = fig.canvas.get_width_height()
                img_data = img_data.reshape((int(img_h), int(img_w), -1))

            return img_data

        return cwt_result

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'wavelet': self.wavelet,
                'scales': self.scales,
                'contourf': self.contourf
            })


# Augmentation transforms
class TimeShiftEEG(EEGTransform):
    def __init__(self, max_shift: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_shift = max_shift

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        shift = random.randint(-self.max_shift, self.max_shift)
        eeg_shifted = np.roll(eeg, shift, axis=-1)  # Time shift along the last axis (time)
        return eeg_shifted

class TimeStretchEEG(EEGTransform):
    def __init__(self, stretch_factor: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stretch_factor = stretch_factor

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        num_samples = eeg.shape[-1]
        new_length = int(num_samples * self.stretch_factor)
        eeg_stretched = np.zeros((eeg.shape[0], new_length))  # Initialize the stretched signal array
        for i, channel in enumerate(eeg):
            eeg_stretched[i] = np.interp(np.linspace(0, num_samples, new_length),
                                         np.arange(num_samples),
                                         channel)
        return eeg_stretched

class RandomCropPadEEG(EEGTransform):
    def __init__(self, target_length: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_length = target_length

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        num_samples = eeg.shape[-1]
        if num_samples > self.target_length:
            # Random crop
            start = random.randint(0, num_samples - self.target_length)
            eeg_cropped = eeg[:, start:start + self.target_length]
        else:
            # Padding with zeros
            eeg_padded = np.pad(eeg, ((0, 0), (0, self.target_length - num_samples)), 'constant')
            return eeg_padded
        return eeg_cropped

class GaussianNoiseEEG(EEGTransform):
    def __init__(self, noise_factor: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_factor = noise_factor

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        noise = np.random.normal(0, self.noise_factor, eeg.shape)
        eeg_noisy = eeg + noise
        return eeg_noisy

class FrequencyMaskingEEG(EEGTransform):
    def __init__(self, freq_mask_param: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq_mask_param = freq_mask_param

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        num_channels, num_samples = eeg.shape
        # Generate frequency mask
        freq_mask_size = int(self.freq_mask_param * num_samples)
        mask_start = random.randint(0, num_samples - freq_mask_size)
        
        # Mask out a portion of the frequency domain (simulating noise or missing data)
        eeg[:, mask_start:mask_start + freq_mask_size] = 0
        return eeg

class ChannelDropoutEEG(EEGTransform):
    def __init__(self, dropout_prob: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_prob = dropout_prob

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        # Apply channel dropout by setting some channels to zero based on dropout probability
        num_channels = eeg.shape[0]
        dropout_mask = np.random.rand(num_channels) < self.dropout_prob
        eeg_dropout = eeg.copy()
        eeg_dropout[dropout_mask] = 0
        return eeg_dropout

class MixupEEG(EEGTransform):
    def __init__(self, alpha: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        # Choose a random sample from the batch for mixup
        batch_size = eeg.shape[0]
        lambda_ = np.random.beta(self.alpha, self.alpha)  # Mixup coefficient

        # Randomly pick another sample to mix with
        idx = random.randint(0, batch_size - 1)
        eeg_mixed = lambda_ * eeg + (1 - lambda_) * eeg[idx]

        return eeg_mixed

class BandPassFilterEEG(EEGTransform):
    def __init__(self, low_freq: float = 0.5, high_freq: float = 50, fs: float = 256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = fs

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        nyquist = 0.5 * self.fs
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist

        b, a = signal.butter(4, [low, high], btype='band')
        eeg_filtered = np.array([signal.filtfilt(b, a, channel) for channel in eeg])

        return eeg_filtered





if __name__ == "__main__":
    # Instantiate the transform                                  # [batch, channel, freq, time]
    t = STFTSpectrogram(n_fft=128, hop_length=2, contourf=False) # [batch,14, 65, 65]
    # t = STFTSpectrogram(n_fft=128, hop_length=1, contourf=False) # [batch,14, 65, 129]

    # Apply to EEG data (shape [channels, time_points])
    eeg_data = np.random.randn(1,14, 128)  
    spectrogram_data = t(eeg=eeg_data)['eeg']

    print(spectrogram_data.shape)  # Shape will be [32, num_freq_bins, num_time_frames]

    plt.imshow(spectrogram_data[0][0])  # Visualize the spectrogram of the first batch and first channel
    plt.colorbar()
    plt.title('Spectrogram (contourf=True) - Channel 1')
    plt.show()

    import sys
    sys.exit()

    # To visualize a single channel's spectrogram (if contourf=True)
    t = STFTSpectrogram(n_fft=64, hop_length=32, contourf=True)
    spectrogram_image = t(eeg=eeg_data)['eeg']
    spectrogram_image.shape

    plt.imshow(spectrogram_image[0])  # Visualize the spectrogram of the first channel
    plt.colorbar()
    plt.title('Spectrogram (contourf=True) - Channel 1')
    plt.show()


    # EEG data (assumed to be a NumPy array of shape [num_channels, num_samples])
    eeg_data = np.random.randn(14, 1000)  # 14 channels, 1000 samples

    # Initialize the augmentation transforms
    time_shift_transform = TimeShiftEEG(max_shift=10)
    time_stretch_transform = TimeStretchEEG(stretch_factor=1.1)
    gaussian_noise_transform = GaussianNoiseEEG(noise_factor=0.02)
    freq_masking_transform = FrequencyMaskingEEG(freq_mask_param=0.2)
    channel_dropout_transform = ChannelDropoutEEG(dropout_prob=0.2)

    # Apply augmentations
    eeg_augmented_shift = time_shift_transform.apply(eeg=eeg_data)
    eeg_augmented_stretch = time_stretch_transform.apply(eeg=eeg_data)
    eeg_augmented_noise = gaussian_noise_transform.apply(eeg=eeg_data)
    eeg_augmented_freq = freq_masking_transform.apply(eeg=eeg_data)
    eeg_augmented_channel_dropout = channel_dropout_transform.apply(eeg=eeg_data)
