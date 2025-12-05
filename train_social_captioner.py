"""
Train a social-media-focused image captioner on the local Instagram datasets.

Usage (example):
python train_social_captioner.py \
  --dataset-root /home/ritvik/NLP/datasets \
  --output-dir /home/ritvik/NLP/artifacts/social-captioner \
  --epochs 3 --train-batch-size 8 --eval-batch-size 8

This script creates a VisionEncoderDecoderModel using:
  - Encoder: google/vit-base-patch16-224-in21k (Vision Transformer)
  - Decoder: gpt2 (GPT-2 language model)

Expected CSV/image layout:
  datasets/instagram_data/captions_csv.csv
  datasets/instagram_data/img/*.jpg
  datasets/instagram_data2/captions_csv2.csv
  datasets/instagram_data2/img2/*.jpg
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    GPT2Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint


def _build_image_path(base_dir: str, relative_path: str) -> str:
    rel = relative_path.strip()
    if not rel.lower().endswith(".jpg"):
        rel = f"{rel}.jpg"
    return os.path.join(base_dir, rel)


def load_caption_dataframe(dataset_root: str) -> pd.DataFrame:
    """
    Load and merge both instagram datasets. Drops rows with missing captions
    or missing image files.
    """
    sources = [
        (
            os.path.join(dataset_root, "instagram_data", "captions_csv.csv"),
            os.path.join(dataset_root, "instagram_data"),
        ),
        (
            os.path.join(dataset_root, "instagram_data2", "captions_csv2.csv"),
            os.path.join(dataset_root, "instagram_data2"),
        ),
    ]

    frames: List[pd.DataFrame] = []
    for csv_path, base_dir in sources:
        if not os.path.exists(csv_path):
            continue

        if csv_path.endswith("captions_csv.csv"):
            df = pd.read_csv(csv_path)
            path_col, caption_col = "Image File", "Caption"
        else:
            df = pd.read_csv(
                csv_path, header=None, names=["Sr No", "Image File", "Caption"]
            )
            path_col, caption_col = "Image File", "Caption"

        df = df[[path_col, caption_col]].rename(columns={path_col: "image_rel", caption_col: "caption"})
        df["image_path"] = df["image_rel"].apply(lambda p: _build_image_path(base_dir, p))
        df["caption"] = df["caption"].astype(str).str.strip()
        df = df[(df["caption"].str.len() > 0) & (df["image_path"].apply(os.path.exists))]
        frames.append(df[["image_path", "caption"]])

    if not frames:
        raise FileNotFoundError("No caption CSVs found under the provided dataset root.")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return merged


class InstagramCaptionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_processor: AutoImageProcessor,
        tokenizer: GPT2Tokenizer,
        max_target_len: int = 64,
    ):
        self.df = dataframe
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values[0]
        labels = self.tokenizer(
            row["caption"],
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss
        return {"pixel_values": pixel_values, "labels": labels}


@dataclass
class DataCollator:
    """
    Small wrapper to keep pixel_values stacked correctly; we delegate padding to
    default_data_collator which understands -100 label masking.
    """

    processor: AutoImageProcessor

    def __call__(self, features):
        return default_data_collator(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="/home/ritvik/NLP/datasets")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--encoder-model", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--decoder-model", type=str, default="gpt2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--train-split", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from. Use 'auto' to pick the last checkpoint in output-dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Set device for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    captions_df = load_caption_dataframe(args.dataset_root)
    split_idx = int(len(captions_df) * args.train_split)
    train_df = captions_df.iloc[:split_idx].reset_index(drop=True)
    eval_df = captions_df.iloc[split_idx:].reset_index(drop=True)

    # Load tokenizer from GPT-2 decoder model
    tokenizer = GPT2Tokenizer.from_pretrained(args.decoder_model)
    # GPT-2 doesn't have a pad token by default; use eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # Load image processor from ViT encoder model
    image_processor = AutoImageProcessor.from_pretrained(args.encoder_model)

    # Create VisionEncoderDecoderModel from separate encoder and decoder
    print(f"Creating VisionEncoderDecoderModel from:")
    print(f"  Encoder: {args.encoder_model}")
    print(f"  Decoder: {args.decoder_model}")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=args.encoder_model,
        decoder_pretrained_model_name_or_path=args.decoder_model,
    )
    # Move model to GPU if available
    model = model.to(device)

    # Configure the model for image captioning
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.num_beams = args.num_beams
    model.config.max_length = args.max_target_length

    # Enable cross-attention in the decoder
    model.decoder.config.is_decoder = True
    model.decoder.config.add_cross_attention = True

    train_dataset = InstagramCaptionDataset(
        train_df, image_processor, tokenizer, max_target_len=args.max_target_length
    )
    eval_dataset = InstagramCaptionDataset(
        eval_df, image_processor, tokenizer, max_target_len=args.max_target_length
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="steps",
        predict_with_generate=True,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),  # Enable mixed precision training on GPU
        dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory for faster GPU transfer
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=DataCollator(processor=image_processor),
    )

    # Handle checkpoint-based resume logic
    resume_path = None
    if args.resume_from_checkpoint == "":
        print("Starting fresh; ignoring existing checkpoints.")
    elif args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            resume_path = get_last_checkpoint(args.output_dir)
            if resume_path is None:
                print("No checkpoint found in output-dir; starting fresh.")
        else:
            resume_path = args.resume_from_checkpoint
        if resume_path:
            print(f"Resuming from checkpoint: {resume_path}")
    else:
        auto_checkpoint = get_last_checkpoint(args.output_dir)
        if auto_checkpoint:
            print(
                f"Found checkpoint in output-dir, resuming from {auto_checkpoint}. "
                "Pass --resume-from-checkpoint '' to force a fresh start."
            )
            resume_path = auto_checkpoint

    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

