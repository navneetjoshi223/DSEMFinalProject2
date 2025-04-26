import torch
from neural_codec.encodec_wrapper import EncodecWrapper
from transformers import AutoTokenizer, T5ForConditionalGeneration
from diffusion.audio_denoising_diffusion import GaussianDiffusion

def load_model_from_args(args):
    text_encoder = None
    text_tokenizer = None
    if args.get("text_encoder_id", None) is not None:
        text_tokenizer = AutoTokenizer.from_pretrained(args["text_encoder_id"])
        text_encoder = T5ForConditionalGeneration.from_pretrained(args["text_encoder_id"])
        print(f"Loaded text encoder: {args['text_encoder_id']}")

    model = GaussianDiffusion(
        dim=args["dim"],
        depth=args["depth"],
        channels=args["channels"],
        timesteps=args["timesteps"],
        loss_type=args.get("loss_type", "l2"),
        conditional=args.get("conditional", True),
        use_text_conditioning=args.get("use_text_conditioning", True),
        text_encoder=text_encoder,
        text_tokenizer=text_tokenizer,
        scale=args.get("scale", 1.0),
        guidance_weight=args.get("guidance_weight", 1.0),
        sampling_timesteps=args.get("sampling_timesteps", None),
    )

    model.cuda()
    return model