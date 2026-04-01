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
"""TAIHRI Evaluation Wrapper Module."""

__version__ = "1.0.0"
__author__ = "TAIHRI Team"
__email__ = "roboticsx@tencent.com"

from .tasks import TaskType
from .wrapper_kpt2d_3d_human_mllm import TAIHRIKpt2D3DMLLMWrapper

__all__ = [
    "TaskType",
    "TAIHRIKpt2D3DMLLMWrapper",
]
