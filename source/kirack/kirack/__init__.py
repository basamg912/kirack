# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import os

kirack_EXT_DIR = os.path.dirname(os.path.abspath(__file__))
"""Path to the `kirack` extension package."""

kirack_DATA_DIR = os.path.join(kirack_EXT_DIR, "tasks", "environment", "kapex", "data", "kapex")
"""Path to the data directory shipped with the extension."""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *
