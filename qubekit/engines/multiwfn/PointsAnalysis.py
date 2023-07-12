import os
import subprocess

import numpy as np

from .CriticalPoints import SpaceFunctions, MWFN_EXECUTABLE


MWFN_CALC_GRID = '5\n'
MWFN_LOAD_POINTS = '100\n'
MWFN_EXIT = 'q\n'


def calculate_rsf_at_points(
    fchk_path: str,
    points: np.ndarray,
    rsf: SpaceFunctions,
    preserve_files: bool = False
) -> np.ndarray:
    np.savetxt('points.txt', points, header='%d' % points.shape[0], comments='')
    space_function = '%d\n' % rsf.value
    subprocess.run([
        MWFN_EXECUTABLE,
        fchk_path
    ], input = (
        MWFN_CALC_GRID + space_function + 
            MWFN_LOAD_POINTS + 'points.txt\n' + 'points.txt\n' + MWFN_EXIT
    ), text=True, stdout=subprocess.DEVNULL)
    rsf = np.loadtxt('points.txt', skiprows=1)
    if not preserve_files:
        os.remove('points.txt')
    return rsf
