import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

# Constants
ENCODEC_SAMPLING_RATE = 24000
LATENT_SAMPLING_RATE = 75

ALIGNED_DATA_DIR = "persist/data/aligned_librispeech/LibriSpeech/test-clean"

class LibriSpeech(Dataset):
    def __init__(self, split='test', tokenizer=None, max_seq_len=512):
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []

        for root, dirs, files in os.walk(ALIGNED_DATA_DIR):
            trans_files = [f for f in files if f.endswith('.trans.txt')]
            for trans_file in trans_files:
                transcriptions = {}
                trans_path = os.path.join(root, trans_file)
                with open(trans_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            utt_id, text = parts
                            transcriptions[utt_id] = text

                for utt_id, text in transcriptions.items():
                    audio_path = os.path.join(root, f"{utt_id}.flac")
                    if os.path.exists(audio_path):
                        self.data.append((audio_path, text))

        print(f"Loaded {len(self.data)} audio-text pairs from {ALIGNED_DATA_DIR}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, text = self.data[idx]

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != ENCODEC_SAMPLING_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=ENCODEC_SAMPLING_RATE)(waveform)

        if self.tokenizer is not None:
            # Hard truncate the text manually first
            text = text[:512]   # ⬅️ force truncate to first 512 chars to avoid overloading tokenizer
            text_tokens = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_seq_len
            )
        else:
            text_tokens = {"input_ids": torch.zeros(self.max_seq_len, dtype=torch.long)}

        return {
            "waveform": waveform.squeeze(0),  # shape: (T,)
            "text": text,
            "text_tokens": text_tokens["input_ids"].squeeze(0)
        }

    @staticmethod
    def collate_fn(batch):
        """Pad waveforms dynamically to the max length in batch"""
        waveforms = [item["waveform"] for item in batch]
        texts = [item["text"] for item in batch]
        text_tokens = [item["text_tokens"] for item in batch]

        # Find max waveform length
        max_len = max([w.shape[0] for w in waveforms])

        # Pad waveforms
        padded_waveforms = []
        for w in waveforms:
            pad_len = max_len - w.shape[0]
            padded_waveforms.append(torch.nn.functional.pad(w, (0, pad_len)))

        return {
            "waveform": torch.stack(padded_waveforms),
            "text": texts,
            "input_ids": torch.stack(text_tokens),           # key changed
            "attention_mask": (torch.stack(text_tokens) != 0) # key added
        }