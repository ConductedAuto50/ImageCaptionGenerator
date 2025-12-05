"""
Generate social-media-ready captions from a fine-tuned model.

Example:
python infer_social_captioner.py \
  --model-dir /home/ritvik/NLP/artifacts/social-captioner \
  --image /home/ritvik/NLP/datasets/instagram_data/img/insta10.jpg

To use the latest checkpoint automatically:
python infer_social_captioner.py \
  --model-dir /home/ritvik/NLP/artifacts/social-captioner \
  --image /path/to/image.jpg \
  --use-latest-checkpoint
"""

import argparse
import os
import re
from typing import Optional

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


def get_latest_checkpoint(model_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the model directory by step number."""
    if not os.path.isdir(model_dir):
        return None
    
    checkpoint_dirs = []
    for name in os.listdir(model_dir):
        path = os.path.join(model_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            match = re.match(r"checkpoint-(\d+)", name)
            if match:
                step = int(match.group(1))
                checkpoint_dirs.append((step, path))
    
    if not checkpoint_dirs:
        return None
    
    # Return the checkpoint with the highest step number
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    return checkpoint_dirs[0][1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to fine-tuned model directory.")
    parser.add_argument("--image", type=str, required=True, help="Path to image to caption.")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Discourage repetitive tags/words often seen in social posts.",
    )
    parser.add_argument(
        "--use-latest-checkpoint",
        action="store_true",
        help="Automatically use the latest checkpoint in model-dir instead of the final saved model.",
    )
    return parser.parse_args()


def generate_caption(
    model_dir: str,
    image_path: str,
    num_beams: int,
    max_length: int,
    repetition_penalty: float,
    processor_dir: Optional[str] = None,
) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    model.eval()  # Set to evaluation mode
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Image processor may be in a different dir (root) when using checkpoints
    processor_path = processor_dir if processor_dir else model_dir
    processor = AutoImageProcessor.from_pretrained(processor_path)

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    output_ids = model.generate(
        pixel_values,
        num_beams=num_beams,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
    )[0]

    caption = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return caption


def main():
    args = parse_args()
    
    model_dir = args.model_dir
    processor_dir = None  # Use model_dir by default
    
    if args.use_latest_checkpoint:
        latest = get_latest_checkpoint(args.model_dir)
        if latest:
            print(f"Using latest checkpoint: {latest}")
            model_dir = latest
            # Image processor config is in the root model dir, not the checkpoint
            processor_dir = args.model_dir
        else:
            print(f"No checkpoints found in {args.model_dir}, using final saved model.")
    
    caption = generate_caption(
        model_dir=model_dir,
        image_path=args.image,
        num_beams=args.num_beams,
        max_length=args.max_length,
        repetition_penalty=args.repetition_penalty,
        processor_dir=processor_dir,
    )
    print(caption)


if __name__ == "__main__":
    main()

