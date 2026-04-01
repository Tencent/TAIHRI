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

"""
Output parsing utilities for TAIHRI
"""

import json
import re
from typing import Dict, List

# Constants for normalization
BIN_SIZE = 999.0  # Maximum bin value for normalized coordinates
FOCAL_SCALE = 1000.0  # Scale factor for focal length normalization


def parse_prediction(
    text: str, w: int, h: int, task_type: str = "detection"
) -> Dict[str, List]:
    """Parse model output text to extract category-wise predictions.

    Args:
        text: Model output text
        w: Image width
        h: Image height
        task_type: Type of task ("detection", "keypoint", etc.)

    Returns:
        Dictionary with category as key and list of predictions as value
    """
    if task_type == "keypoint":
        return parse_keypoint_prediction(text, w, h)
    elif task_type == "keypoint_3d":
        return parse_keypoint_3d_prediction(text, w, h)
    else:
        return parse_standard_prediction(text, w, h)


def parse_standard_prediction(text: str, w: int, h: int) -> Dict[str, List]:
    """
    Parse standard prediction output for detection, pointing, etc.

    Input format example:
    "<|object_ref_start|>person<|object_ref_end|><|box_start|><0><35><980><987>, <646><0><999><940><|box_end|>"

    Returns:
    {
        'category1': [{"type": "box/point/polygon", "coords": [...]}],
        'category2': [{"type": "box/point/polygon", "coords": [...]}],
        ...
    }
    """
    result = {}

    # Remove the end marker if present
    text = text.split("<|im_end|>")[0]
    if not text.endswith("<|box_end|>"):
        text = text + "<|box_end|>"

    # Use regex to find all object references and coordinate pairs
    pattern = r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>"
    matches = re.findall(pattern, text)

    for category, coords_text in matches:
        category = category.strip()

        # Find all coordinate tokens in the format <{number}>
        coord_pattern = r"<(\d+)>"
        _ = re.findall(coord_pattern, coords_text)

        annotations = []
        # Split by comma to handle multiple coordinates for the same phrase
        coord_strings = coords_text.split(",")

        for coord_str in coord_strings:
            coord_nums = re.findall(coord_pattern, coord_str.strip())

            if len(coord_nums) == 2:
                # Point: <{x}><{y}>
                try:
                    x_bin = int(coord_nums[0])
                    y_bin = int(coord_nums[1])

                    # Convert from bins [0, 999] to absolute coordinates
                    x = (x_bin / BIN_SIZE) * w
                    y = (y_bin / BIN_SIZE) * h

                    annotations.append({"type": "point", "coords": [x, y]})
                except (ValueError, IndexError) as e:
                    print(f"Error parsing point coordinates: {e}")
                    continue

            elif len(coord_nums) == 4:
                # Bounding box: <{x0}><{y0}><{x1}><{y1}>
                try:
                    x0_bin = int(coord_nums[0])
                    y0_bin = int(coord_nums[1])
                    x1_bin = int(coord_nums[2])
                    y1_bin = int(coord_nums[3])

                    # Convert from bins [0, 999] to absolute coordinates
                    x0 = (x0_bin / BIN_SIZE) * w
                    y0 = (y0_bin / BIN_SIZE) * h
                    x1 = (x1_bin / BIN_SIZE) * w
                    y1 = (y1_bin / BIN_SIZE) * h

                    annotations.append({"type": "box", "coords": [x0, y0, x1, y1]})
                except (ValueError, IndexError) as e:
                    print(f"Error parsing box coordinates: {e}")
                    continue

            elif len(coord_nums) > 4 and len(coord_nums) % 2 == 0:
                # Polygon: <{x0}><{y0}><{x1}><{y1}>...
                try:
                    polygon_coords = []
                    for i in range(0, len(coord_nums), 2):
                        x_bin = int(coord_nums[i])
                        y_bin = int(coord_nums[i + 1])

                        # Convert from bins [0, 999] to absolute coordinates
                        x = (x_bin / BIN_SIZE) * w
                        y = (y_bin / BIN_SIZE) * h

                        polygon_coords.append([x, y])

                    annotations.append({"type": "polygon", "coords": polygon_coords})
                except (ValueError, IndexError) as e:
                    print(f"Error parsing polygon coordinates: {e}")
                    continue

        if category not in result:
            result[category] = []
        result[category].extend(annotations)

    return result


def _convert_bbox_coords(bbox: str, instance_id: str, w: int, h: int) -> Optional[List[float]]:
    """Convert bbox coordinates from bins to absolute coordinates."""
    if not isinstance(bbox, str) or not bbox.strip():
        print(f"Invalid bbox format for {instance_id}: {bbox}")
        return None

    coord_pattern = r"<(\d+)>"
    coord_matches = re.findall(coord_pattern, bbox)

    if len(coord_matches) != 4:
        print(
            f"Invalid bbox format for {instance_id}: expected 4 coordinates, "
            f"got {len(coord_matches)}"
        )
        return None

    try:
        x0_bin, y0_bin, x1_bin, y1_bin = [int(match) for match in coord_matches]
        x0 = (x0_bin / BIN_SIZE) * w
        y0 = (y0_bin / BIN_SIZE) * h
        x1 = (x1_bin / BIN_SIZE) * w
        y1 = (y1_bin / BIN_SIZE) * h
        return [x0, y0, x1, y1]
    except (ValueError, IndexError) as e:
        print(f"Error parsing bbox coordinates: {e}")
        return None


def _convert_keypoint_coords(kp_coords: Any, kp_name: str, w: int, h: int) -> Any:
    """Convert keypoint coordinates from bins to absolute coordinates."""
    if kp_coords == "unvisible" or kp_coords is None:
        return "unvisible"

    if not isinstance(kp_coords, str) or not kp_coords.strip():
        return "unvisible"

    coord_pattern = r"<(\d+)>"
    coord_matches = re.findall(coord_pattern, kp_coords)

    if len(coord_matches) != 2:
        print(
            f"Invalid keypoint format for {kp_name}: expected 2 coordinates, "
            f"got {len(coord_matches)}"
        )
        return "unvisible"

    try:
        x_bin, y_bin = [int(match) for match in coord_matches]
        x = (x_bin / BIN_SIZE) * w
        y = (y_bin / BIN_SIZE) * h
        return [x, y]
    except (ValueError, IndexError) as e:
        print(f"Error parsing keypoint coordinates for {kp_name}: {e}")
        return "unvisible"


def _extract_json_from_text(text: str) -> str:
    """Extract JSON content from text with markdown code blocks or plain JSON."""
    json_pattern = r"```json\s*(.*?)\s*```"
    json_matches = re.findall(json_pattern, text, re.DOTALL)

    if not json_matches:
        # Try to find JSON without markdown
        try:
            # Look for JSON-like structure
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                return text[start_idx : end_idx + 1]
        except Exception:  # pylint: disable=broad-except
            pass
        return ""

    return json_matches[0]


def _convert_3d_keypoint_coords(
    kp_coords: Any,
    kp_name: str,
    xy_range: Tuple[float, float],
    z_range: Tuple[float, float]
) -> Any:
    """Convert 3D keypoint coordinates from bins to unnormalized coordinates."""
    if not isinstance(kp_coords, str) or not kp_coords.strip():
        return "unvisible"

    coord_pattern = r"<(\d+)>"
    coord_matches = re.findall(coord_pattern, kp_coords)

    if len(coord_matches) != 3:
        print(
            f"Invalid 3D keypoint format for {kp_name}: expected 3 coordinates, "
            f"got {len(coord_matches)}"
        )
        return "unvisible"

    try:
        x_bin, y_bin, z_bin = [int(match) for match in coord_matches]
        x_unnormalized = x_bin / BIN_SIZE * (xy_range[1] - xy_range[0]) + xy_range[0]
        y_unnormalized = y_bin / BIN_SIZE * (xy_range[1] - xy_range[0]) + xy_range[0]
        z_unnormalized = z_bin / BIN_SIZE * (z_range[1] - z_range[0]) + z_range[0]
        return [x_unnormalized, y_unnormalized, z_unnormalized]
    except (ValueError, IndexError) as e:
        print(f"Error parsing 3D keypoint coordinates for {kp_name}: {e}")
        return "unvisible"


def parse_keypoint_prediction(text: str, w: int, h: int) -> Dict[str, List]:
    """
    Parse keypoint task JSON output to extract bbox and keypoints.

    Expected format:
    ```json
    {
        "person1": {
            "bbox": " <1> <36> <987> <984> ",
            "keypoints": {
                "nose": " <540> <351> ",
                "left eye": " <559> <316> ",
                "right eye": "unvisible",
                ...
            }
        },
        ...
    }
    ```

    Returns:
    Dict with category as key and list of keypoint instances as value
    """
    # Extract JSON content from text
    json_str = _extract_json_from_text(text)
    if not json_str:
        return {}

    try:
        keypoint_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing keypoint JSON: {e}")
        return {}

    result = {}

    for instance_id, instance_data in keypoint_data.items():
        if "bbox" not in instance_data or "keypoints" not in instance_data:
            continue

        bbox = instance_data["bbox"]
        keypoints = instance_data["keypoints"]

        # Convert bbox coordinates
        converted_bbox = _convert_bbox_coords(bbox, instance_id, w, h)
        if converted_bbox is None:
            continue

        # Convert keypoint coordinates
        converted_keypoints = {}
        for kp_name, kp_coords in keypoints.items():
            converted_keypoints[kp_name] = _convert_keypoint_coords(
                kp_coords, kp_name, w, h
            )

        # Extract category from instance_id
        category = "keypoint_instance"
        if instance_id:
            category_match = re.match(r"^([a-zA-Z_]+)", instance_id)
            if category_match:
                category = category_match.group(1)

        if category not in result:
            result[category] = []

        result[category].append(
            {
                "type": "keypoint",
                "bbox": converted_bbox,
                "keypoints": converted_keypoints,
                "instance_id": instance_id,
            }
        )

    return result


def parse_keypoint_2d_3d_prediction(
    text: str,
    w: int,
    h: int,
    xy_range: Tuple[float, float],
    z_range: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Parse 3D keypoint task JSON output to extract 3D keypoints.

    Expected format:
    ```json
    {
        "keypoints_2d": {
            "spine2": "<770><465>",
            "pelvis": "<783><507>",
            "left_knee": "<797><594>",
            "right_elbow": "<709><502>",
            "left_wrist": "<841><424>",
            "left_collar": "<773><443>"
        }
        "keypoints_3d": {
            "spine2": "<770><465><760>",
            "pelvis": "<783><507><773>",
            "left_knee": "<797><594><750>",
            "right_elbow": "<709><502><714>",
            "left_wrist": "<841><424><697>",
            "left_collar": "<773><443><739>"
        }
    }
    ```

    Args:
        text: Model output text
        w: Image width
        h: Image height
        xy_range: Range for XY coordinates
        z_range: Range for Z coordinates

    Returns:
        Dict with category as key and list of 3D keypoint instances as value
    """
    # Extract JSON content from markdown code blocks
    json_pattern = r"```json\s*(.*?)\s*```"
    json_matches = re.findall(json_pattern, text, re.DOTALL)

    if not json_matches:
        # Try to find JSON without markdown
        try:
            # Look for JSON-like structure
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx : end_idx + 1]
            else:
                return {}
        except Exception:  # pylint: disable=broad-except
            return {}
    else:
        json_str = json_matches[0]

    # matches = list(re.finditer(r'"keypoints_3d"\s*:', json_str))

    # # 如果出现 >=2，则替换第一个
    # if len(matches) >= 2:
    #     json_str = re.sub(r'"keypoints_3d"\s*:', '"keypoints_2d":', json_str, count=1)

    try:
        keypoint_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing 3D keypoint JSON: {e}")
        return {}

    result = {}

    if "keypoints_2d" not in keypoint_data:
        result["keypoints_2d"] = {}

    else:
        keypoints_2d = keypoint_data["keypoints_2d"]

        converted_keypoints_2d = {}
        for kp_name, kp_coords in keypoints_2d.items():
            if isinstance(kp_coords, str) and kp_coords.strip():
                # Parse box tokens from string format like " <770><465><760> "
                coord_pattern = r"<(\d+)>"
                coord_matches = re.findall(coord_pattern, kp_coords)

                if len(coord_matches) == 2:
                    try:
                        x_bin, y_bin = [int(match) for match in coord_matches]
                        x_unnormalized = x_bin / BIN_SIZE * w
                        y_unnormalized = y_bin / BIN_SIZE * h
                        converted_keypoints_2d[kp_name] = [x_unnormalized, y_unnormalized]
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing 2D keypoint coordinates for {kp_name}: {e}")
                        converted_keypoints_2d[kp_name] = "unvisible"
                else:
                    print(
                        f"Invalid 2D keypoint format for {kp_name}: expected 2 coordinates, got {len(coord_matches)}"
                    )
                    converted_keypoints_2d[kp_name] = "unvisible"
            else:
                converted_keypoints_2d[kp_name] = "unvisible"

        result["keypoints_2d"] = converted_keypoints_2d

    if "keypoints_3d" not in keypoint_data:
        result["keypoints_3d"] = {}

    else:
        keypoints_3d = keypoint_data["keypoints_3d"]

        converted_keypoints_3d = {}
        for kp_name, kp_coords in keypoints_3d.items():
            converted_keypoints_3d[kp_name] = _convert_3d_keypoint_coords(
                kp_coords, kp_name, xy_range, z_range
            )

        result["keypoints_3d"] = converted_keypoints_3d

    return result


def parse_keypoint_2d_3d_tokenizer_prediction(
    text: str,
    w: int,
    h: int,
    xy_range: Tuple[float, float],
    z_range: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Parse 3D keypoint task JSON output to extract 3D keypoints.

    Expected format:
    format: <|kpt2d_start|> <|left_shoulder|> <261>, <630> <|right_shoulder|> <145>, <627>
            <|left_elbow|> <290>, <748> <|right_elbow|> <125>, <725> <|kpt2d_end|>
            <|kpt3d_start|> <|left_shoulder|> <394>, <559>, <416> <|right_shoulder|> <330>, <562>, <444>
            <|left_elbow|> <404>, <613>, <421> <|right_elbow|> <314>, <612>, <456> <|kpt3d_end|>

    Args:
        text: Model output text
        w: Image width
        h: Image height
        xy_range: Range for XY coordinates
        z_range: Range for Z coordinates

    Returns:
        Dict with parsed keypoints
    """
    result = {}

    bbox_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"
    bbox_match = re.search(bbox_pattern, text, re.DOTALL)

    if bbox_match:
        bbox_content = bbox_match.group(1)
        # Parse box tokens from string format like " <1> <36> <987> <984> "
        coord_pattern = r"<(\d+)>"
        coord_matches = re.findall(coord_pattern, bbox_content)

        if len(coord_matches) == 4:
            try:
                x0_bin, y0_bin, x1_bin, y1_bin = [
                    int(match) for match in coord_matches
                ]
                x0 = (x0_bin / BIN_SIZE) * w
                y0 = (y0_bin / BIN_SIZE) * h
                x1 = (x1_bin / BIN_SIZE) * w
                y1 = (y1_bin / BIN_SIZE) * h
                converted_bbox = [x0, y0, x1, y1]
                result["bbox"] = converted_bbox
            except (ValueError, IndexError) as e:
                print(f"Error parsing bbox coordinates: {e}")

    # 2D Keypoints
    kpt2d_pattern = r"<\|kpt2d_start\|>(.*?)<\|kpt2d_end\|>"
    kpt2d_match = re.search(kpt2d_pattern, text, re.DOTALL)

    converted_keypoints_2d = {}
    if kpt2d_match:
        kpt2d_content = kpt2d_match.group(1)
        # Support both name formats and optional commas: <|name|> <x>, <y> or <name><x><y>
        pattern_2d = r"(?:<\|(.*?)\|>|<(?!\d+>)(.*?)>)\s*<(\d+)>\s*(?:,\s*)?<(\d+)>"
        for m in re.finditer(pattern_2d, kpt2d_content, re.DOTALL):
            kp_name = m.group(1) if m.group(1) is not None else m.group(2)
            try:
                x_bin = int(m.group(3))
                y_bin = int(m.group(4))
                x_unnormalized = x_bin / BIN_SIZE * w
                y_unnormalized = y_bin / BIN_SIZE * h
                converted_keypoints_2d[(kp_name or '').strip()] = [x_unnormalized, y_unnormalized]
            except (ValueError, IndexError) as e:
                print(f"Error parsing 2D keypoint coordinates for {kp_name}: {e}")
                converted_keypoints_2d[(kp_name or '').strip()] = "unvisible"

    result["keypoints_2d"] = converted_keypoints_2d

    # 3D Keypoints
    kpt3d_pattern = r"<\|kpt3d_start\|>(.*?)<\|kpt3d_end\|>"
    kpt3d_match = re.search(kpt3d_pattern, text, re.DOTALL)

    converted_keypoints_3d = {}
    if kpt3d_match:
        kpt3d_content = kpt3d_match.group(1)
        # Support both name formats and optional commas: <|name|> <x>, <y>, <z> or <name><x><y><z>
        pattern_3d = r"(?:<\|(.*?)\|>|<(?!\d+>)(.*?)>)\s*<(\d+)>\s*(?:,\s*)?<(\d+)>\s*(?:,\s*)?<(\d+)>"
        for m in re.finditer(pattern_3d, kpt3d_content, re.DOTALL):
            kp_name = m.group(1) if m.group(1) is not None else m.group(2)
            try:
                x_bin = int(m.group(3))
                y_bin = int(m.group(4))
                z_bin = int(m.group(5))
                x_unnormalized = x_bin / BIN_SIZE * (xy_range[1] - xy_range[0]) + xy_range[0]
                y_unnormalized = y_bin / BIN_SIZE * (xy_range[1] - xy_range[0]) + xy_range[0]
                z_unnormalized = z_bin / BIN_SIZE * (z_range[1] - z_range[0]) + z_range[0]
                converted_keypoints_3d[(kp_name or '').strip()] = [x_unnormalized, y_unnormalized, z_unnormalized]
            except (ValueError, IndexError) as e:
                print(f"Error parsing 3D keypoint coordinates for {kp_name}: {e}")
                converted_keypoints_3d[(kp_name or '').strip()] = "unvisible"
    result["keypoints_3d"] = converted_keypoints_3d
    return result


def parse_keypoint_3d_prediction(
    text: str,
    w: int,
    h: int,
    xy_range: Tuple[float, float],
    z_range: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Parse 3D keypoint task JSON output to extract 3D keypoints.

    Expected format:
    ```json
    {
        "keypoints": {
            "spine2": "<770><465><760>",
            "pelvis": "<783><507><773>",
            "left_knee": "<797><594><750>",
            "right_elbow": "<709><502><714>",
            "left_wrist": "<841><424><697>",
            "left_collar": "<773><443><739>"
        }
    }
    ```

    Args:
        text: Model output text
        w: Image width
        h: Image height
        xy_range: Range for XY coordinates
        z_range: Range for Z coordinates

    Returns:
        Dict with category as key and list of 3D keypoint instances as value
    """
    # Extract JSON content from text
    json_str = _extract_json_from_text(text)
    if not json_str:
        return {}

    try:
        keypoint_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing 3D keypoint JSON: {e}")
        return {}

    result = {}

    if "keypoints" not in keypoint_data:
        return {}

    keypoints = keypoint_data["keypoints"]

    converted_keypoints = {}
    for kp_name, kp_coords in keypoints.items():
        converted_keypoints[kp_name] = _convert_3d_keypoint_coords(
            kp_coords, kp_name, xy_range, z_range
        )

    result["keypoints_3d"] = converted_keypoints

    return result


def convert_boxes_to_normalized_bins(
    boxes: List[List[float]], ori_width: int, ori_height: int
) -> List[str]:
    """Convert boxes from absolute coordinates to normalized bins (0-999) and map to words."""
    word_mapped_boxes = []
    for box in boxes:
        x0, y0, x1, y1 = box

        # Normalize coordinates to [0, 1] range
        x0_norm = max(0.0, min(1.0, x0 / ori_width))
        x1_norm = max(0.0, min(1.0, x1 / ori_width))
        y0_norm = max(0.0, min(1.0, y0 / ori_height))
        y1_norm = max(0.0, min(1.0, y1 / ori_height))

        # Convert to bins [0, 999]
        x0_bin = max(0, min(999, int(x0_norm * 999)))
        y0_bin = max(0, min(999, int(y0_norm * 999)))
        x1_bin = max(0, min(999, int(x1_norm * 999)))
        y1_bin = max(0, min(999, int(y1_norm * 999)))

        # Map to words
        word_mapped_box = "".join(
            [
                f"<{x0_bin}>",
                f"<{y0_bin}>",
                f"<{x1_bin}>",
                f"<{y1_bin}>",
            ]
        )
        word_mapped_boxes.append(word_mapped_box)

    return word_mapped_boxes
