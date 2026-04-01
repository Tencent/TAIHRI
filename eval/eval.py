#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Tencent. All rights reserved.
#
# This file is part of TAIHRI.
#
# TAIHRI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TAIHRI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with TAIHRI.  If not, see <https://www.gnu.org/licenses/>.
#
"""TAIHRI Evaluation Script."""

# Add parent directory to path (must be before other imports)
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Standard library imports
import argparse
from typing import List, Tuple

# Third-party imports
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Local imports
from eval_wrapper import TAIHRIKpt2D3DMLLMWrapper

# Constants for normalization
FOCAL_SCALE = 1000.0  # Scale factor for focal length normalization
BATCH_SIZE = 64  # Default batch size for processing images


def parse_args():
    parser = argparse.ArgumentParser(description="TAIHRI Demo")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Model path or HuggingFace repo ID",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["transformers", "vllm"],
        help="Backend to use for inference",
    )
    parser.add_argument("--focal_length", type=float, default=600.0)
    parser.add_argument("--quantization", type=str, default="awq")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=2560 * 28 * 28)
    parser.add_argument('--input_path_dir', type=str, default='')  # Path to input image or directory
    parser.add_argument('--annot_path', type=str, default='')  # Path to annotation file
    parser.add_argument('--output_path', type=str, default='')  # Path to save output image
    parser.add_argument(
        '--test_dataset',
        type=str,
        default='harmony',
        help="Which dataset to test on: harmony, egobody, or spec"
    )
    parser.add_argument(
        '--part',
        type=str,
        default='upper_body',
        help='Which part to process: upper_body or lower_body'
    )
    args = parser.parse_args()
    return args


def egobody_dataloader(annot_file: str) -> List[str]:
    """Load image paths from egobody annotation file.

    Args:
        annot_file: Path to annotation .npy file

    Returns:
        List of image file paths
    """
    if not os.path.exists(annot_file):
        raise FileNotFoundError(f"Annotation file not found: {annot_file}")

    data = np.load(annot_file, allow_pickle=True)
    img_path_list = []

    for img_path in data['image_path']:
        # Extract last 5 parts of path and reconstruct
        path_parts = img_path.split('/')[-5:]
        img_path = os.path.join('finetuning', 'data', 'img', *path_parts)
        img_path_list.append(img_path)

    return img_path_list


def spec_dataloader(annot_file: str) -> Tuple[List[str], List[float]]:
    """Load image paths and focal lengths from SPEC annotation file.

    Args:
        annot_file: Path to annotation .npy file

    Returns:
        Tuple of (image_paths, focal_lengths)
    """
    if not os.path.exists(annot_file):
        raise FileNotFoundError(f"Annotation file not found: {annot_file}")

    data = np.load(annot_file, allow_pickle=True)
    img_path_list = []
    focal_list = []

    for idx, img_path in enumerate(data['image_path']):
        img_path = os.path.join('finetuning', 'data', 'img', img_path.replace('imgs', 'spec_mtp_jpg'))
        focal_list.append(float(data['focal_length'][idx]))
        img_path_list.append(img_path)

    return img_path_list, focal_list


def harmony_dataloader(input_path_dir: str) -> List[List[str]]:
    """Load image paths from harmony dataset annotation files.

    Args:
        input_path_dir: Path to directory containing annotation files

    Returns:
        List of image path lists for different body parts
    """
    annot_files = [
        os.path.join(input_path_dir, f'harmony4d_test_{name}_center.npy')
        for name in ["leftarm", "rightarm", "upperbody"]
    ]
    all_annots = []
    for annot_file in annot_files:
        data = np.load(annot_file, allow_pickle=True)
        img_path_list = []
        for img_path in list(data.item().keys()):
            img_path = os.path.join(os.path.dirname(__file__), '..', 'data', img_path)
            img_path_list.append(img_path)
        all_annots.append(img_path_list)
    return all_annots


def crop_batch_images(
    images: List[Image.Image],
    crop_size: Tuple[int, int],
    focal_length: Union[float, List[float], np.ndarray],
    crop: bool = True
) -> List[Image.Image]:
    """Crop images to the given crop_size (width, height).

    Args:
        images: List of PIL images to crop
        crop_size: Target crop size (width, height)
        focal_length: Focal length(s) for scaling
        crop: Whether to crop the images

    Returns:
        List of cropped images
    """

    def process_single_image(img: Image.Image, focal: float) -> Image.Image:
        """Process a single image: resize and optionally crop."""
        width, height = img.size
        scale_factor = FOCAL_SCALE / focal
        img = img.resize((
            int(width * scale_factor),
            int(height * scale_factor)
        ), Image.LANCZOS)
        if crop:
            img_width, img_height = img.size
            crop_width = min(crop_size[0], img_width)
            crop_height = min(crop_size[1], img_height)
            left = (img_width - crop_width) // 2
            top = (img_height - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            img = img.crop((left, top, right, bottom))
        return img

    cropped_images = []
    # Assume images are the same size
    if isinstance(focal_length, (list, np.ndarray)):
        for img, focal in zip(images, focal_length):
            cropped_images.append(process_single_image(img, focal[0]))
    else:
        for img in images:
            cropped_images.append(process_single_image(img, focal_length))
    return cropped_images

if __name__ == "__main__":
    args = parse_args()
    input_path_dir = args.input_path_dir
    output_path = args.output_path
    test_dataset = args.test_dataset
    os.makedirs(output_path, exist_ok=True)

    print("🚀 Initializing TAIHRI model...")
    print(f"Model: {args.model_path}")
    print(f"Backend: {args.backend}")

    # Initialize TAIHRI model
    taihri_model = TAIHRIKpt2D3DMLLMWrapper(
        model_path=args.model_path,
        backend=args.backend,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        quantization=args.quantization,
    )

    dataloader_map = {
        "harmony": lambda: harmony_dataloader(input_path_dir),
        "egobody": lambda: egobody_dataloader(args.annot_path),
        "spec": lambda: spec_dataloader(args.annot_path),
    }

    def process_images_in_batches(
        img_paths: List[str],
        keypoint_ids: List[int],
        focal_length: Any,
        crop: bool = True
    ) -> Dict:
        """Process images in batches and return keypoint predictions."""
        kpt3d_dict = {}
        num_batches = (len(img_paths) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in tqdm(range(0, len(img_paths), BATCH_SIZE), desc="Processing images", total=num_batches):
            batch = img_paths[i : i + BATCH_SIZE]
            img_list = []
            for path in batch:
                image = Image.open(path).convert("RGB")
                img_list.append(image)

            # Handle focal_length for batch
            batch_focal = focal_length
            if isinstance(focal_length, list):
                batch_focal = focal_length[i : i + BATCH_SIZE]

            img_list = crop_batch_images(
                img_list,
                crop_size=(1400, 1400),
                focal_length=batch_focal,
                crop=crop
            )
            with torch.no_grad():
                results = taihri_model.inference(
                    images=img_list,
                    task="keypoint_2d_3d",
                    keypoint_ids=keypoint_ids
                )
            for idx, result in enumerate(results):
                kpt3d = result['extracted_predictions']['keypoints_3d']
                kpt3d_dict[img_paths[i + idx]] = kpt3d
        return kpt3d_dict

    if test_dataset == 'harmony':
        all_img_paths = dataloader_map[test_dataset]()
        print("✅ Model initialized successfully!")

        # indices_list = [[1,2,4,5], [16, 18, 20], [17, 19, 21], [16, 17, 18, 19]]
        # example keypoint indices to extract
        indices_list = [[16, 18, 20], [17, 19, 21], [16, 17, 18, 19]]
        # indices_list = [[16, 17, 18, 19]]  # example keypoint indices to extract
        # part_names = ["lower_body", "left_arm", "right_arm", "upper_body"]
        part_names = ["left_arm", "right_arm", "upper_body"]
        for path_id, img_paths in enumerate(all_img_paths):
            print(f"Processing {len(img_paths)} images...")
            kpt3d_dict = process_images_in_batches(
                img_paths, indices_list[path_id], args.focal_length, crop=True
            )
            np.savez(os.path.join(output_path, f"{part_names[path_id]}.npz"), **kpt3d_dict)
    elif test_dataset == 'egobody':
        img_paths = dataloader_map[test_dataset]()
        print(f"Processing {len(img_paths)} images...")
        print("✅ Model initialized successfully!")

        if args.part == "upper_body":
            indices_list = [[16, 18, 20], [17, 19, 21], [16, 17, 18, 19]]  # example keypoint indices to extract
            part_names = ["left_arm", "right_arm", "upper_body"]
        elif args.part == "lower_body":
            indices_list = [[1, 2, 4, 5]]  # example keypoint indices to extract
            part_names = ["lower_body"]
        else:
            raise ValueError("Invalid part name. Choose from 'upper_body' or 'lower_body'.")
        for idxx, indices in enumerate(indices_list):
            kpt3d_dict = process_images_in_batches(
                img_paths, indices, args.focal_length, crop=False
            )
            np.savez(os.path.join(output_path, f"egobody_{part_names[idxx]}.npz"), **kpt3d_dict)
    elif test_dataset == 'spec':
        img_paths, focal_list = dataloader_map[test_dataset]()
        print(f"Processing {len(img_paths)} images...")
        print("✅ Model initialized successfully!")

        indices_list = [[16, 18, 20], [17, 19, 21], [16, 17, 18, 19], [1, 2, 4, 5]]
        # example keypoint indices to extract
        part_names = ["left_arm", "right_arm", "upper_body", "lower_body"]
        for idxx, indices in enumerate(indices_list):
            kpt3d_dict = process_images_in_batches(
                img_paths, indices, focal_list, crop=False
            )
            np.savez(os.path.join(output_path, f"spec_{part_names[idxx]}.npz"), **kpt3d_dict)
    else:
        raise ValueError("Invalid test_dataset. Choose from 'harmony', 'egobody', 'spec'.")
