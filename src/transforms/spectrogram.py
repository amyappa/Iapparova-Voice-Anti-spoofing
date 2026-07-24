import torch
import torch.nn.functional as F
import torchaudio


class SpectrogramTransform(torch.nn.Module):
    def __init__(
        self, n_fft=512, n_frames=750, win_length=320, hop_length=160, random_crop=True
    ):
        super().__init__()

        self.n_frames = n_frames
        self.random_crop = random_crop

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=320,
            hop_length=160,
            window_fn=torch.blackman_window,
            normalized=False,
            center=False,
        )

    def forward(self, waveform):
        spectrogram = self.spectrogram(waveform)
        spectrogram = torch.log(spectrogram + 1e-10)

        frame_count = spectrogram.shape[-1]

        if frame_count < self.n_frames:
            frames_to_add = self.n_frames - frame_count
            spectrogram = F.pad(spectrogram, (0, frames_to_add))

        elif frame_count > self.n_frames:
            if self.random_crop:
                start_frame = torch.randint(
                    frame_count - self.n_frames + 1, (1,)
                ).item()
            else:
                start_frame = (frame_count - self.n_frames) // 2

            spectrogram = spectrogram[:, start_frame : start_frame + self.n_frames]

        return spectrogram.unsqueeze(0)
