import re
import json
import loader
import numpy as np

from collections import defaultdict
from typing import Dict, List


def read_uploaded_files(files: List, pattern: str) -> Dict[str, np.ndarray]:
    """ Reads uploaded files (UploadedFile) with pattern grouping."""
    if not files:
        return []

    dataset = defaultdict(dict)
    for file in files:
        groups = re.match(pattern, file.name).groupdict()
        model, classname = groups["model"], groups["classname"]
        dataset[model][classname] = json.load(file)

    datastructures = dict()
    for model, data in dataset.items():
        datastructures[model] = loader.read_structures(data)
    return dict(datastructures)
