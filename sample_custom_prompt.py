import torch
import os
import json
from diffusion.audio_denoising_diffusion import GaussianDiffusion
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
ckpt_dir = "checkpoints/simple-tts-pretrained"
model_path = os.path.join(ckpt_dir, "model.pt")
args_path = os.path.join(ckpt_dir, "args.json")

# Load model args
with open(args_path) as f:
    model_args = json.load(f)

# Build model
diffusion = GaussianDiffusion(
    dim=model_args["dim"],
    dim_mults=model_args["dim_mults"],
    num_transformer_layers=model_args["num_transformer_layers"],
    dropout=model_args["dropout"],
    scale_skip_connection=model_args["scale_skip_connection"],
    inpainting_embedding=model_args["inpainting_embedding"],
    conformer_transformer=model_args["conformer_transformer"],
    objective=model_args["objective"],
    parameterization=model_args["parameterization"],
    loss_type=model_args["loss_type"],
    ddpm_var=model_args["ddpm_var"],
    sampler=model_args["sampler"]
)

# Load weights
state_dict = torch.load(model_path, map_location=device)
diffusion.load_state_dict(state_dict, strict=False)
diffusion.to(device).eval()

# Prompt
prompt = "The quick brown fox jumps over the lazy dog."

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_args["text_encoder"])
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Sampling
samples = diffusion.sample(
    cond_input=input_ids,
    batch_size=1,
    guidance_scale=model_args.get("scale", 1.0),
    num_timesteps=model_args.get("sampling_timesteps", 250)
)

# Save sample
os.makedirs("outputs", exist_ok=True)
torch.save(samples.cpu(), "outputs/sample_audio.pt")
print("Sample saved to outputs/sample_audio.pt")