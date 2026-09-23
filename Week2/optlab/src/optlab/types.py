"""Type aliases used throughout optlab. PROVIDED — do not edit."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Vec: TypeAlias = npt.NDArray[np.float64]
Mat: TypeAlias = npt.NDArray[np.float64]
Index: TypeAlias = npt.NDArray[np.intp]
