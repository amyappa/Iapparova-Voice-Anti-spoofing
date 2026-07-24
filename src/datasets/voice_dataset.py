from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset


class VoiceDataset(Dataset):
    def __init__(self, protocol, audio_directory):
        self.protocol = Path(protocol)
        self.audio_directory = Path(audio_directory)
        self.items = []
        with self.protocol.open("r") as file:
            for line in file:
                segments = line.strip().split()

                if segments[4] == "bonafide":
                    label = 0
                else:
                    label = 1
                self.items.append({"audio_id": segments[1], "label": label})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        audio_id = item["audio_id"]
        audio_path = self.audio_directory / f"{audio_id}.flac"
        waveform, sample_rate = sf.read(audio_path, dtype="float32")
        waveform = torch.from_numpy(waveform)
        return {
            "audio_id": audio_id,
            "data_object": waveform,
            "labels": item["label"],
        }
