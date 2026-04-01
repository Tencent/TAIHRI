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
"""TAIHRI Demo Script."""

# Add parent directory to path (must be before other imports)
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Standard library imports
import argparse
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Local imports
from eval_wrapper import TAIHRIKpt2D3DMLLMWrapper

# Constants for normalization
FOCAL_SCALE = 1000.0  # Scale factor for focal length normalization

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
    parser.add_argument(
        "--adapter_model_path",
        default="",
        help="peft lora model",
    )
    parser.add_argument("--focal_length", type=float, default=600.0)
    parser.add_argument("--princpt_x", type=float, default=704.0)
    parser.add_argument("--princpt_y", type=float, default=704.0)
    parser.add_argument("--quantization", type=str, default="awq")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=2560 * 28 * 28)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=1234)

    parser.add_argument('--input_path', type=str, default='')  # Path to input image or directory
    parser.add_argument('--output_path', type=str, default='')  # Path to save output image
    parser.add_argument('--prompts', type=str, default='Give the person a big hug.', help='Prompts for human interaction')
    parser.add_argument('--crop', action='store_true', help='Whether to crop the input images to square.')
    parser.add_argument('--write_coordinates', action='store_true', help='Whether to write 3D coordinates on the output image.')
    args = parser.parse_args()
    return args

def simple_dataloader(img_dir: str) -> List[str]:
    """Load image file paths from directory.

    Args:
        img_dir: Path to directory containing images

    Returns:
        List of image file paths
    """
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Input directory not found: {img_dir}")

    image_files = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])

    if not image_files:
        raise ValueError(f"No valid image files found in directory: {img_dir}")

    return image_files

def visualize_2d_3d_keypoints(
    image: Image.Image,
    keypoints_3d: Optional[Dict[str, List[float]]],
    cam_intrinsics: Union[Dict[str, float], np.ndarray],
    output_path: str,
    bbox: Optional[List[float]] = None,
    write_coords: bool = True
) -> None:
    """Visualize 3D keypoints on image using OpenCV.

    Args:
        image: PIL Image to draw on
        keypoints_3d: 3D keypoints dictionary {name: [x, y, z]}
        cam_intrinsics: Camera intrinsic parameters
        output_path: Path to save output image
        bbox: Optional bounding box [x1, y1, x2, y2]
        write_coords: Whether to write coordinates on image
    """


    # Convert PIL image to numpy (BGR for cv2)
    img_np = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Camera intrinsics
    if isinstance(cam_intrinsics, dict):
        K = np.array([
            [cam_intrinsics['fx'], 0, cam_intrinsics['cx']],
            [0, cam_intrinsics['fy'], cam_intrinsics['cy']],
            [0, 0, 1]
        ])
    else:
        K = np.array(cam_intrinsics)

    if bbox is not None:
        # Draw bbox
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw 3d keypoints
    if keypoints_3d is not None:
        img_h, img_w = img_cv.shape[:2]

        # 根据图像大小自适应的字号和粗细
        base_size = max(img_w, img_h)
        font_scale = max(0.6, min(2.5, base_size / 800.0))
        thickness = max(1, int(round(font_scale * 2)))

        font = cv2.FONT_HERSHEY_SIMPLEX
        label_boxes = []  # 已放置文字的包围框，用于避免重叠

        def is_overlapping(x1, y1, x2, y2):
            for bx1, by1, bx2, by2 in label_boxes:
                if not (x2 < bx1 or bx2 < x1 or y2 < by1 or by2 < y1):
                    return True
            return False

        for name, xyz in keypoints_3d.items():
            xyz = np.array(xyz).reshape(3, 1)
            uvw = K @ xyz
            if uvw[2, 0] == 0:
                continue
            u = int(round(uvw[0, 0] / uvw[2, 0]))
            v = int(round(uvw[1, 0] / uvw[2, 0]))

            # Draw circle
            cv2.circle(img_cv, (u, v), 12, (0, 0, 150), -1)

            if not write_coords:
                continue
            # 文字内容
            text = f"{name}:({xyz[0, 0]:.2f}, {xyz[1, 0]:.2f}, {xyz[2, 0]:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # 首选位置：关键点右上方
            base_x = u + 5
            base_y = v - 5

            # 在若干候选位置中寻找不重叠的位置
            x = base_x
            y = base_y
            placed = False
            for _ in range(15):
                # 保证在图像范围内
                x_clamped = max(0, min(img_w - text_w - 1, x))
                y_clamped = max(text_h + 1, min(img_h - 1, y))
                x1, y1, x2, y2 = x_clamped, y_clamped - text_h, x_clamped + text_w, y_clamped

                if not is_overlapping(x1, y1, x2, y2):
                    x, y = x_clamped, y_clamped
                    placed = True
                    break

                # 先往上移，移不动就往下
                if y - (text_h + 4) > text_h:
                    y -= (text_h + 4)
                else:
                    y = v + text_h + 5

            if not placed:
                # 实在找不到合适位置就放在关键点正下方
                x = max(0, min(img_w - text_w - 1, u - text_w // 2))
                y = max(text_h + 1, min(img_h - 1, v + text_h + 5))
                x1, y1, x2, y2 = x, y - text_h, x + text_w, y

            # 记录文字包围框
            label_boxes.append((x1, y1, x2, y2))

            # Draw label
            cv2.putText(
                img_cv,
                text,
                (x, y),
                font,
                font_scale,
                (255, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
    cv2.imwrite(output_path, img_cv)  # Save for debugging

def crop_batch_images(images: List[Image.Image], crop_size: Tuple[int, int],
                      focal_length: float, crop: bool = False) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]], List[float]]:
    """Crop and resize batch of images.

    Args:
        images: List of PIL Images
        crop_size: Target crop size (width, height)
        focal_length: Camera focal length for scaling
        crop: Whether to crop images

    Returns:
        Tuple of (cropped_images, crop_boxes, scale_factors)

    Raises:
        ValueError: If images list is empty or invalid parameters
    """
    if not images:
        raise ValueError("Images list cannot be empty")

    try:
        width, height = images[0].size
        scale_factor = FOCAL_SCALE / float(focal_length)

        cropped_images = []
        crop_boxes = []
        scale_factors = []

        for img in images:
            # Resize image
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            if new_width <= 0 or new_height <= 0:
                raise ValueError(f"Invalid scaled dimensions: {new_width}x{new_height}")

            img = img.resize((new_width, new_height), Image.LANCZOS)
            img_width, img_height = img.size

            # Calculate crop region
            if crop:
                crop_width = min(crop_size[0], img_width)
                crop_height = min(crop_size[1], img_height)
                left = (img_width - crop_width) // 2
                top = (img_height - crop_height) // 2
                right = left + crop_width
                bottom = top + crop_height
            else:
                left, top, right, bottom = 0, 0, img_width, img_height

            # Validate crop coordinates
            if left >= right or top >= bottom:
                raise ValueError(f"Invalid crop coordinates: ({left},{top},{right},{bottom})")

            cropped_img = img.crop((left, top, right, bottom))
            cropped_images.append(cropped_img)
            crop_boxes.append((left, top, right, bottom))
            scale_factors.append(scale_factor)

        return cropped_images, crop_boxes, scale_factors

    except Exception as e:
        raise RuntimeError(f"Failed to crop batch images: {str(e)}") from e

def unproj(
    keypoints_2d: Dict[str, List[float]],
    bbox: List[float],
    crop_box: Tuple[int, int, int, int],
    scale_factor: float
) -> Tuple[Dict[str, List[float]], List[float]]:
    """Unproject 2D keypoints to original image coordinates.

    Args:
        keypoints_2d: Dictionary of 2D keypoints
        bbox: Bounding box coordinates [x0, y0, x1, y1]
        crop_box: Crop box coordinates (left, top, right, bottom)
        scale_factor: Scale factor for unprojection

    Returns:
        Tuple of (unprojected_keypoints, unprojected_bbox)
    """
    left, top, right, bottom = crop_box
    unproj_keypoints = {}
    bbox = [
        (bbox[0] + left) / scale_factor,
        (bbox[1] + top) / scale_factor,
        (bbox[2] + left) / scale_factor,
        (bbox[3] + top) / scale_factor
    ]
    for name, uv in keypoints_2d.items():
        u = (uv[0] + left) / scale_factor
        v = (uv[1] + top) / scale_factor
        unproj_keypoints[name] = [u, v]
    return unproj_keypoints, bbox



if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path
    prompts = args.prompts

    print("🚀 Initializing TAIHRI model...")
    print(f"Model: {args.model_path}")
    print(f"Backend: {args.backend}")

    # Initialize TAIHRI model
    taihri_model = TAIHRIKpt2D3DMLLMWrapper(
        model_path=args.model_path,
        adapter_model_path=args.adapter_model_path,
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
    cam_intrinsics = torch.tensor([
        [args.focal_length, 0.0, args.princpt_x],
        [0.0, args.focal_length, args.princpt_y],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    print("✅ Model initialized successfully!")
    images = simple_dataloader(input_path)
    batch_size = 4
    print(f"Processing {len(images)} images...")
    results_list = []
    raw_images = []
    crop_box_list = []
    scale_factors = []
    num_batches = (len(images) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(images), batch_size), desc="Processing images", total=num_batches):
        batch = images[i:i + batch_size]
        img_list = []
        for path in batch:
            image = Image.open(path).convert("RGB")
            img_list.append(image)
        raw_images += img_list
        img_list, crop_boxes, batch_scale_factors = crop_batch_images(img_list, crop_size=(1400, 1400), focal_length=args.focal_length, crop=args.crop)
        crop_box_list += crop_boxes
        scale_factors += batch_scale_factors
        # cropped_images += img_list
        with torch.no_grad():
            results = taihri_model.inference(
                images=img_list,
                task="human_prompt",
                # keypoint_ids=[16, 17, 18, 19],
                prompts = args.prompts,
            )
        results_list += results
    assert len(results_list) == len(images), "Number of results does not match number of images"
    cam_intrinsics = {
        'fx': float(cam_intrinsics[0,0]),
        'fy': float(cam_intrinsics[1,1]),
        'cx': float(cam_intrinsics[0,2]),
        'cy': float(cam_intrinsics[1,2])
    }
    '''
    'extracted_predictions':
     {
        'keypoints_2d':
            {
                'left_wrist': [-0.12262262262262258, 0.007507507507507505],
                'right_wrist': [-0.007507507507507505, -0.012512512512512508]
            }
    }
    {
        'keypoints_3d':
            {
                'left_wrist': [-0.12262262262262258, 0.007507507507507505, 2.5145145145145147],
                'right_wrist': [-0.007507507507507505, -0.012512512512512508, 2.314314314314314]
            }
    }
    '''
    os.makedirs(output_path, exist_ok=True)
    for idx, res in enumerate(results_list):
        print(f"--- Image {idx} ---")
        if res.get("success", False):
            predictions = res["extracted_predictions"]
            bbox = predictions.get("bbox", [0,0,0,0])
            keypoints_2d = predictions.get("keypoints_2d", {})
            keypoints_2d, bbox = unproj(keypoints_2d, bbox, crop_box_list[idx], scale_factors[idx])
            print("Bbox:")
            print(f"  {bbox}")
            print("2D Keypoints:")
            for kpt_name, coords in keypoints_2d.items():
                print(f"  {kpt_name}: {coords}")
            keypoints_3d = predictions.get("keypoints_3d", {})
            print("3D Keypoints:")
            for kpt_name, coords in keypoints_3d.items():
                print(f"  {kpt_name}: {coords}")
            # Visualize and save output image
            output_image_path = os.path.join(output_path, f"output_{idx}.png")
            visualize_2d_3d_keypoints(
                raw_images[idx], keypoints_3d, cam_intrinsics, output_image_path, bbox=bbox, write_coords=True
            )
            print(f"Visualization saved to {output_image_path}")
        else:
            print(f"Inference failed: {res.get('error', 'Unknown error')}")
