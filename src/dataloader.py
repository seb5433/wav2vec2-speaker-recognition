import os
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import tensor, float32, long
import soundfile as sf
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, csv_path, label_dict, audio_base_path='./data/audios'):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"The CSV path {csv_path} does not exist.")
        if not os.path.exists(audio_base_path):
            raise FileNotFoundError(f"The base audio path {audio_base_path} does not exist.")

        self.data = pd.read_csv(csv_path)
        self.label_dict = label_dict
        self.audio_base_path = audio_base_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_base_path, self.data.loc[idx, 'file'])
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"The audio file {audio_path} does not exist.")

        arr = np.zeros(3)

        arr[self.label_dict[self.data.loc[idx, 'speaker']]] = 1
        try:
            speech_array, _ = sf.read(audio_path)
        except Exception as e:
            raise IOError(f"An error occurred when reading the audio file: {e}")

        return tensor(speech_array, dtype=float32), tensor(arr, dtype=long)

    def collate_fn(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        sequences, labels = zip(*batch)
        sequences_padded = pad_sequence(sequences, batch_first=True)
        sequences_labels = pad_sequence(labels, batch_first=True)
        return sequences_padded, sequences_labels
