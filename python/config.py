from typing import List

# Collect and use only next CAD layers for matching.
layerslist: List[str] = ["I-WALL", "A-WALL", "A-GLAZ", "A-DOOR", "A-DOOR-FRAM"]

# Use C++ macthing implementation instead of the Python one.
use_cpp_matching: bool = False
