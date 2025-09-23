"""
This module initializes the hmordd package and sets up the project paths.
"""

import os
import sys
from pathlib import Path

# Define the root of the project.
# This assumes the script is located in HMORDD/src/python/hmordd
ROOT_DIR = Path(__file__).parent.parent.parent.parent

class Paths:
    """A class to manage the paths of the project."""
    root = ROOT_DIR
    resources = ROOT_DIR / "resources"
    outputs = ROOT_DIR / "outputs"
    results = ROOT_DIR / "results"
    
    bin = resources / "bin"
    instances = resources / "instances"
    
    dds = outputs / "dds"
    sols = outputs / "sols"
    
    
# Add the bin directory to the python path to find the C++ binaries
sys.path.append(str(Paths.bin / "setpacking"))
sys.path.append(str(Paths.bin / "knapsack"))
sys.path.append(str(Paths.bin / "tsp"))
