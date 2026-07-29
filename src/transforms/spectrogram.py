import torch
import torch.nn.functional as F
import torchaudio


class SpectrogramTransform(torch.nn.Module):
    def __init__(self, n_fft=1724, n_frames=600, win_length=1724, hop_length=130):
        super().__init__()

        self.n_frames = n_frames

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=torch.blackman_window,
            normalized=False,
            center=False,
        )

    def forward(self, waveform):
        spectrogram = self.spectrogram(waveform)

        frame_count = spectrogram.shape[-1]

        if frame_count < self.n_frames:
            frames_to_add = self.n_frames - frame_count
            spectrogram = F.pad(
                spectrogram,
                (0, frames_to_add),
                value=0.0,
            )

        elif frame_count > self.n_frames:
            spectrogram = spectrogram[..., : self.n_frames]

        spectrogram = torch.log(spectrogram + 1e-10)

        return spectrogram.unsqueeze(0)
