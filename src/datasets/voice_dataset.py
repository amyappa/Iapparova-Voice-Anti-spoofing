from pathlib import Path

import soundfile as sf
import torch

from src.datasets.base_dataset import BaseDataset


class VoiceDataset(BaseDataset):
    def __init__(
        self,
        protocol,
        audio_directory,
        *args,
        **kwargs,
    ):
        protocol = Path(protocol)
        audio_directory = Path(audio_directory)
        items = []

        with protocol.open("r") as file:
            for line in file:
                segments = line.strip().split()

                audio_id = segments[1]
                if segments[4] == "bonafide":
                    label = 0
                else:
                    label = 1

                audio_path = audio_directory / f"{audio_id}.flac"

                items.append({"path": str(audio_path), "label": label})

        super().__init__(
            index=items,
            *args,
            **kwargs,
        )

    def load_object(self, path):
        waveform, sample_rate = sf.read(
            path,
            dtype="float32",
        )
        return torch.from_numpy(waveform)
