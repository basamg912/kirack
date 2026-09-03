
"""Package containing asset and sensor configurations."""

import os
import toml

##
# Configuration for different assets.
##

# Conveniences to other module directories via relative paths
KAPEX_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
"""Path to the extension source directory."""

# KAPEX_DATA_DIR = os.path.join(KAPEX_EXT_DIR, "data")
"""Path to the extension data directory."""

# KAPEX_METADATA = toml.load(os.path.join(KAPEX_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
# __version__ = KAPEX_METADATA["package"]["version"]


from .kapex0 import *
