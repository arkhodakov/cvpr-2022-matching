from typing import List

""" Collect and use only next CAD layers for matching."""
layerslist: List[str] = ["I-WALL", "A-WALL", "A-GLAZ", "A-DOOR", "A-DOOR-FRAM"]

""" Enable Numba & CUDA optimized algorithms.
    All of the algorithms are defined in `calculations.py`."""
enable_optimization: bool = True

""" Normalize user's predictions with ground-truth no matter the initial vertices parameters are.
    Normalization includes rotation, scale, translation, ratios approximation.
    In the result data outputted all the differences will be specified."""
enable_normalization: bool = True

""" Accuracy thresholds specified in ground-truth units.
    The maximum is also used as LAP threshold value where the L2 norm is the default measurement."""
accuracy_thresholds: List[float] = [0.05, 0.10, 0.20]
