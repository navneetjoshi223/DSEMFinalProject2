import os
import shutil

# Dummy audio file you recorded
dummy_audio = "audio_0.wav"  # Change if your file has a different name

# Where you want to copy
base_path = "saved_models/mls/sample16/samples/step_0"
guide_versions = ["guide2.0", "guide3.0", "guide5.0"]

# How many samples
num_samples = 16

for guide in guide_versions:
    guide_path = os.path.join(base_path, guide)
    for idx in range(num_samples):
        dest_path = os.path.join(guide_path, f"audio_{idx}.wav")
        shutil.copy(dummy_audio, dest_path)
        print(f"Copied {dummy_audio} to {dest_path}")