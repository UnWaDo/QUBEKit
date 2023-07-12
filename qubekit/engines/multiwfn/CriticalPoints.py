from enum import Enum
import os
import subprocess
from typing import List, Tuple, Union
import numpy as np


BOHR_TO_ANGSTROM = 0.529177


class Elements(Enum):
    H = 1
    C = 6
    N = 7
    O = 8
    F = 9
    S = 16
    Cl = 17
    Br = 35
    I = 53


MWFN_TOPOLOGY = '2\n'
MWFN_SEL_FUN = '-11\n'
MWFN_FROM_NUCLEI = '2\n'
MWFN_FROM_MIDPOINT = '3\n'
MWFN_FROM_TRIANGLE = '4\n'
MWFN_FROM_PYRAMID = '5\n'
MWFN_FROM_SPHERE = '6\n'
MWFN_SPHERE_RAD = '10\n'
MWFN_SPHERE_POINTS = '11\n'
MWFN_SPHERE_SET_CENTER = '1\n'
MWFN_SPHERE_SOME_NUCLEI = '-2\n'
MWFN_SPHERE_ALL_NUCLEI = '-1\n'
MWFN_SPHERE_START = '0\n'
MWFN_SPHERE_RETURN = '-9\n'
MWFN_CPS_EXPORT = '-4\n'
MWFN_CPS_SAVE = '4\n'
MWFN_CPS_RETURN = '0\n'
MWFN_RETURN = '-10\n'
MWFN_EXIT = 'q\n'

MWFN_EXECUTABLE = '/home/lalex/QC/Multiwfn_3.8/Multiwfn'


class SpaceFunctions(Enum):
    Laplasian = 3
    ELF = 9
    LOL = 10
    ESP = 12


class CritPoints(Enum):
    p3m3 = 1
    p3m1 = 2
    p3p1 = 3
    p3p3 = 4


def generate_molecular_cps(
    fchk_path: str,
    rsf: SpaceFunctions,
    atoms: Union[None, List[Elements]] = None
) -> Tuple[List[CritPoints], List[np.ndarray]]:
    if atoms is None:
        interesting_nuclei = MWFN_SPHERE_ALL_NUCLEI
    else:
        interesting_nuclei = MWFN_SPHERE_SOME_NUCLEI + \
            ','.join([str(i + 1) for i, a in enumerate(atoms) if a > Elements.C.value]) + '\n'
    space_function = '%d\n' % rsf.value
    if os.path.exists('CPs.txt'):
        os.remove('CPs.txt')
    subprocess.run([
        MWFN_EXECUTABLE,
        fchk_path
    ], input = (
        MWFN_TOPOLOGY + MWFN_SEL_FUN + space_function + 
            MWFN_FROM_NUCLEI + MWFN_FROM_MIDPOINT + MWFN_FROM_TRIANGLE + MWFN_FROM_PYRAMID +
            MWFN_FROM_SPHERE + MWFN_SPHERE_RAD + '6\n' + MWFN_SPHERE_POINTS + '5000\n' +
                interesting_nuclei + MWFN_SPHERE_RETURN +
            MWFN_CPS_EXPORT +  MWFN_CPS_SAVE + MWFN_CPS_RETURN + MWFN_RETURN + MWFN_EXIT
    ), text=True, stdout=subprocess.DEVNULL)
    with open('CPs.txt', 'r') as file:
        cps = file.readlines()[1:]
        cps = [l.split() for l in cps]
        cps_type = [CritPoints(int(c[-1])) for c in cps]
        cps_coords = [np.array([float(v) * BOHR_TO_ANGSTROM for v in c[1:-1]]) for c in cps]
    os.remove('CPs.txt')
    return cps_type, cps_coords


def generate_atomic_cps(
    fchk_path: str,
    rsf: SpaceFunctions,
    atoms_i: List[int]
) -> Tuple[List[CritPoints], List[np.ndarray]]:
    nuclei = MWFN_SPHERE_SOME_NUCLEI + ','.join([str(i + 1) for i in atoms_i]) + '\n'
    space_function = '%d\n' % rsf.value
    if os.path.exists('CPs.txt'):
        os.remove('CPs.txt')
    proc = subprocess.run([
        MWFN_EXECUTABLE,
        fchk_path
    ], input = (
        MWFN_TOPOLOGY + MWFN_SEL_FUN + space_function +
            MWFN_FROM_SPHERE + MWFN_SPHERE_RAD + '3\n' + MWFN_SPHERE_POINTS + '5000\n' +
                nuclei + MWFN_SPHERE_RETURN +
            MWFN_CPS_EXPORT +  MWFN_CPS_SAVE + MWFN_CPS_RETURN + MWFN_RETURN + MWFN_EXIT
    ), text=True, stdout=subprocess.PIPE)
    with open('CPs.txt', 'r') as file:
        cps = file.readlines()[1:]
        cps = [l.split() for l in cps]
        cps_type = [CritPoints(int(c[-1])) for c in cps]
        cps_coords = [np.array([float(v) * BOHR_TO_ANGSTROM for v in c[1:-1]]) for c in cps]
    os.remove('CPs.txt')
    return cps_type, cps_coords


def generate_point_cps(
    fchk_path: str,
    rsf: SpaceFunctions,
    coordinate: np.ndarray
) -> Tuple[List[CritPoints], List[np.ndarray]]:
    point = MWFN_SPHERE_SET_CENTER + ','.join([str(c) for c in coordinate]) + '\n'
    space_function = '%d\n' % rsf.value
    if os.path.exists('CPs.txt'):
        os.remove('CPs.txt')
    subprocess.run([
        MWFN_EXECUTABLE,
        fchk_path
    ], input = (
        MWFN_TOPOLOGY + MWFN_SEL_FUN + space_function +
            MWFN_FROM_SPHERE + MWFN_SPHERE_RAD + '3\n' + MWFN_SPHERE_POINTS + '5000\n' +
                point + MWFN_SPHERE_START + MWFN_SPHERE_RETURN +
            MWFN_CPS_EXPORT +  MWFN_CPS_SAVE + MWFN_CPS_RETURN + MWFN_RETURN + MWFN_EXIT
    ), text=True, stdout=subprocess.DEVNULL)
    with open('CPs.txt', 'r') as file:
        cps = file.readlines()[1:]
        cps = [l.split() for l in cps]
        cps_type = [CritPoints(int(c[-1])) for c in cps]
        cps_coords = [np.array([float(v) * BOHR_TO_ANGSTROM for v in c[1:-1]]) for c in cps]
    os.remove('CPs.txt')
    return cps_type, cps_coords


def generate_xyz(
    atoms: Tuple[List[Elements], List[np.ndarray]],
    cps: Tuple[List[CritPoints], List[np.ndarray]],
    name: str = ''
) -> str:
    xyz = ['%d\n%s' % (len(atoms[0]) + len(cps[0]), name)]
    for i, element in enumerate(atoms[0]):
        xyz.append('%2s % 20.8e % 20.8e % 20.8e' % (
            element.name,
            atoms[1][i][0],
            atoms[1][i][1],
            atoms[1][i][2]
        ))
    ATOMS = {CritPoints.p3m3: 'He', CritPoints.p3m1: 'Ne', CritPoints.p3p1: 'Ar', CritPoints.p3p3: 'Kr'}
    for i, cp_type in enumerate(cps[0]):
        xyz.append('%2s % 20.8e % 20.8e % 20.8e' % (
            ATOMS[cp_type],
            cps[1][i][0],
            cps[1][i][1],
            cps[1][i][2]
        ))
    xyz.append('')
    return '\n'.join(xyz)

def save_cps(
    cps: Tuple[List[CritPoints], List[np.ndarray]],
    filename: str
):
    lines = []
    for i, coord in enumerate(cps[1]):
        lines.append('%2d % 20.8e % 20.8e % 20.8e\n' % (
            cps[0][i].value,
            coord[0],
            coord[1],
            coord[2]
        ))
    with open(filename, 'w+') as f:
        f.writelines(lines)

def read_cps(
    filename: str
) -> Tuple[List[CritPoints], List[np.ndarray]]:
    cps = ([], [])
    with open(filename, 'r') as f:
        for line in f:
            values = line.split()
            cps[0].append(CritPoints(int(values[0])))
            cps[1].append(np.array([
                float(values[1]),
                float(values[2]),
                float(values[3])
            ]))
    return cps
