import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import scipy
import os
import json
from glob import glob

class AudioMNIST_CLIPDataset(Dataset):
    """
    AudioMNIST_CLIPDataset
    ----------------------
    Loads the AudioMNIST dataset from Dataset/Data/ and returns (datapoint, prompt) tuples
    for CLIP-style training (audio ↔ text).

    Example output:
        (
            torch.Tensor of shape [2, 128, 44],
            "A 30-year-old male German speaker saying the digit 7."
        )
    """

    def __init__(
        self,
        root="./Dataset",
        meta_file="audioMNIST_meta.txt",
        target_sample_rate=12000,
        duration=None,
        transform=None,
        reshapeDimensions=(2,128,125)
    ):
        super(AudioMNIST_CLIPDataset, self).__init__()
        self.root = root
        self.meta_file = os.path.join(root, meta_file)
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        self.transform = transform
        self.reshapeDimensions = reshapeDimensions
        self.n_fft = 512

        # Load JSON metadata
        with open(self.meta_file, 'r') as f:
            self.meta_json = json.load(f)

        # Build (file_path, prompt) pairs
        self.samples = self._build_samples()

    def _build_samples(self):
        """Collect all wav files and attach a prompt using metadata."""
        samples = []
        speaker_dirs = sorted(os.listdir(self.root))

        for spk in speaker_dirs:
            spk_path = os.path.join(self.root, spk)
            if not os.path.isdir(spk_path):
                continue

            # Get metadata if available
            meta = self.meta_json.get(spk, {})
            gender = meta.get("gender", "unknown")
            age = meta.get("age", "unknown")
            accent = meta.get("accent", "unknown")

            # Find all wav files for this speaker
            wavs = glob(os.path.join(spk_path, "*.wav"))
            for wav_path in wavs:
                # Label = first character in filename (digit 0–9)
                digit = os.path.basename(wav_path)[0]
                prompt = f"A {age}-year-old {gender} {accent} speaker saying the digit {digit}."
                samples.append((wav_path, prompt))
        return samples

    def get_spectrogram(self, waveform):
        """Compute normalized complex STFT spectrogram."""
        f, t, Zxx = scipy.signal.stft(
            waveform,
            self.target_sample_rate,
            nperseg=self.n_fft // 2,
            noverlap=self.n_fft // 4,
            window='hann'
        )
        Zxx = Zxx[0:128, :-1]
        amplitudes = np.abs(Zxx)
        phases = np.angle(Zxx)
        amp_min, amp_max = np.min(amplitudes), np.max(amplitudes)
        eps = 1e-10
        normalized_amplitudes = (amplitudes - amp_min) / (amp_max - amp_min + eps)
        Zxx_normalized = normalized_amplitudes * np.exp(1j * phases)
        return Zxx_normalized

    def complex_to_2d(self, tensor):
        """Split a complex tensor into two channels: real and imaginary."""
        new_tensor = np.zeros((2, tensor.shape[0]), dtype=np.float64)
        new_tensor[0] = np.real(tensor)
        new_tensor[1] = np.imag(tensor)
        return new_tensor

    def __getitem__(self, index):
        """Return (audio_tensor, prompt) tuple with auto padding."""
        file_path, prompt = self.samples[index]
        waveform, _ = librosa.load(file_path, sr=self.target_sample_rate, duration=self.duration)
        if self.transform:
            waveform = self.transform(waveform)

        Zxx = self.get_spectrogram(waveform)
        datapoint = self.complex_to_2d(Zxx.flatten())

        # --- SAFE reshape + padding/cropping ---
        freq_bins = 128
        total_values = datapoint.shape[1]
        time_frames = total_values // freq_bins  # compute how many frames exist

        if time_frames == 0:
            # Edge case: extremely short waveform
            datapoint = np.zeros((2, freq_bins, 44))
        else:
            # Trim to full frames only
            datapoint = datapoint[:, :freq_bins * time_frames]
            datapoint = datapoint.reshape(2, freq_bins, time_frames)

            # Pad or crop to fixed 44 time frames
            if time_frames < 44:
                pad_width = 44 - time_frames
                datapoint = np.pad(datapoint, ((0, 0), (0, 0), (0, pad_width)))
            else:
                datapoint = datapoint[:, :, :44]

        return torch.tensor(datapoint, dtype=torch.float32), prompt

    def __len__(self):
        return len(self.samples)
