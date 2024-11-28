from typing import Dict, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from torcheeg.transforms.base_transform import EEGTransform
import io 

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



if __name__ == "__main__":
    # Instantiate the transform
    t = STFTSpectrogram(n_fft=64, hop_length=32, contourf=False)

    # Apply to EEG data (shape [channels, time_points])
    eeg_data = np.random.randn(14, 128)  # 32 channels, 1000 time points
    spectrogram_data = t(eeg=eeg_data)['eeg']

    print(spectrogram_data.shape)  # Shape will be [32, num_freq_bins, num_time_frames]



    # To visualize a single channel's spectrogram (if contourf=True)
    t = STFTSpectrogram(n_fft=64, hop_length=32, contourf=True)
    spectrogram_image = t(eeg=eeg_data)['eeg']
    spectrogram_image.shape

    plt.imshow(spectrogram_image[0])  # Visualize the spectrogram of the first channel
    plt.colorbar()
    plt.title('Spectrogram (contourf=True) - Channel 1')
    plt.show()