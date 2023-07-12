import os
from typing import List, Tuple

from .la_selectors import *
from qubekit.engines.multiwfn import CritPoints, SpaceFunctions, Elements
from qubekit.engines.multiwfn import generate_atomic_cps, generate_point_cps, generate_xyz
from qubekit.engines.multiwfn import save_cps, read_cps

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from rdkit.Chem.rdchem import Atom as RDAtom


TYPES = {
    'ELF': SpaceFunctions.ELF,
    'LOL': SpaceFunctions.LOL,
    'Lap': SpaceFunctions.Laplasian
}
CHALCOGENS = [Elements.O, Elements.S]
PNICTOGENS = [Elements.N]
HALOGENS = [Elements.F, Elements.Cl, Elements.Br, Elements.I]


def select_atomic_points(
    fchk_path: str,
    mol: RDMol,
    atom_index: int,
    mode_rsf: SpaceFunctions,
    mode_number: int,
    mode_fluorine: bool = False,
    reuse = False
) -> List[np.ndarray]:
    atom = mol.GetAtomWithIdx(atom_index)
    element = Elements(atom.GetAtomicNum())
    if not mode_fluorine and element == Elements.F:
        return []
    connected = atom.GetNeighbors()
    conformer = mol.GetConformers()[0]
    if len(connected) >= 3 and are_coplanar(
        [conformer.GetAtomPosition(atom.GetIdx())] +
            [conformer.GetAtomPosition(a.GetIdx()) for a in connected]
    ):
        return []
    mode_cp_type = None
    cp_function = None
    if mode_rsf in [SpaceFunctions.ELF, SpaceFunctions.LOL]:
        if element in CHALCOGENS:
            if mode_number == 1:
                mode_cp_type = [CritPoints.p3m1, CritPoints.p3p1]
                cp_function = select_one_on_reverse
            elif mode_number == 2:
                mode_cp_type = [CritPoints.p3m3]
                if len(connected) == 2:
                    cp_function = select_two_tetrahedral
                elif len(connected) == 1:
                    cp_function = select_two_on_120
            elif mode_number == 3:
                if len(connected) == 2:
                    mode_cp_type = [CritPoints.p3m1, CritPoints.p3p1]
                    cp_function = select_one_on_reverse
                elif len(connected) == 1:
                    mode_cp_type = [CritPoints.p3m3]
                    cp_function = select_two_on_120
        elif element in PNICTOGENS:
            mode_cp_type = [CritPoints.p3m3]
            if len(connected) == 2:
                cp_function = select_one_on_reverse
            elif len(connected) == 3:
                cp_function = select_one_on_reverse
        elif element in HALOGENS:
            mode_cp_type = [CritPoints.p3p1]
            cp_function = select_one_on_reverse
    elif mode_rsf == SpaceFunctions.Laplasian:
        if element in CHALCOGENS:
            if mode_number == 1:
                cp_function = select_one_on_reverse
                if len(connected) == 2:
                    mode_cp_type = [CritPoints.p3m1]
                elif len(connected) == 1:
                    mode_cp_type = [CritPoints.p3m3]
            elif mode_number == 2:
                mode_cp_type = [CritPoints.p3p1]
                if len(connected) == 2:
                    cp_function = select_two_tetrahedral
                elif len(connected) == 1:
                    cp_function = select_two_on_120
            elif mode_number == 3:
                if len(connected) == 2:
                    mode_cp_type = [CritPoints.p3m1]
                    cp_function = select_one_on_reverse
                elif len(connected) == 1:
                    mode_cp_type = [CritPoints.p3p1]
                    cp_function = select_two_on_120
        elif element in PNICTOGENS:
            if len(connected) == 2:
                mode_cp_type = [CritPoints.p3m1]
                cp_function = select_one_on_reverse
            elif len(connected) == 3:
                mode_cp_type = [CritPoints.p3p1]
                cp_function = select_one_on_reverse
        elif element in HALOGENS:
            mode_cp_type = [CritPoints.p3m3]
            cp_function = select_one_on_reverse
    if mode_cp_type is None or cp_function is None:
        raise Exception('Invalid RSF mode: %s (found type? %r. found function? %r)' % (
            '_'.join([mode_rsf.name, str(mode_number), 'F' if mode_fluorine else 'X']),
            mode_cp_type is not None,
            cp_function is not None
        ))
    name, _ = os.path.splitext(os.path.basename(fchk_path))
    name = '%s_%d_%s%s.cps' % (name, atom.GetIdx(), mode_rsf.name, 'F' if mode_fluorine else '')
    if reuse and os.path.exists(name):
        all_cps = read_cps(name)
    else:
        all_cps = generate_atomic_cps(fchk_path, mode_rsf, [atom.GetIdx()])
        if reuse:
            save_cps(all_cps, name)
    cps = select_points_by_type(all_cps, mode_cp_type)
    cps_coordinates = cp_function(
        conformer.GetAtomPosition(atom.GetIdx()),
        [conformer.GetAtomPosition(a.GetIdx()) for a in connected],
        cps
    )
    return list(cps_coordinates)


def select_aromatic_points(
    fchk_path: str,
    mol: RDMol,
    cycle: List[int],
    mode_rsf: SpaceFunctions,
    reuse = False
) -> List[np.ndarray]:
    conformer = mol.GetConformers()[0]
    cycle_coords = [conformer.GetAtomPosition(i) for i in cycle]
    if not are_coplanar(cycle_coords):
        return []
    center = sum(cycle_coords, np.zeros(3)) / len(cycle_coords)
    name, _ = os.path.splitext(os.path.basename(fchk_path))
    name = '%s_%s_%s.cps' % (name, '_'.join([str(c) for c in cycle]), mode_rsf.name)
    if reuse and os.path.exists(name):
        all_cps = read_cps(name)
    else:
        all_cps = generate_point_cps(
            fchk_path,
            mode_rsf,
            center
        )
        if reuse:
            save_cps(all_cps, name)
    cps_coordinates = select_one_at_center(center, all_cps[1])
    return list(cps_coordinates)

def gen_oscs(
    fchk_file: str,
    mol: RDMol,
    mode_rsf: str,
    mode_number: int,
    mode_fluorine: bool,
    reuse = False
) -> Tuple[List[Tuple[int, np.ndarray]], List[Tuple[List[int], np.ndarray]]]:
    mode_rsf = TYPES[mode_rsf]
    atom_oscs = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() <= 6:
            continue
        a_points = select_atomic_points(
            fchk_path = fchk_file,
            mol = mol,
            atom = a,
            mode_rsf = mode_rsf,
            mode_number = mode_number,
            mode_fluorine = mode_fluorine,
            reuse = reuse
        )
        atom_oscs.extend([(a.GetIdx(), ap) for ap in a_points])
    ring_oscs = []
    for cycle in Chem.GetSSSR(mol):
        crit_point = select_aromatic_points(
            fchk_path = fchk_file,
            mol = mol,
            cycle = cycle,
            mode_rsf = mode_rsf,
            reuse = reuse
        )
        if len(crit_point):
            ring_oscs.append(([c for c in cycle], crit_point[0]))
    return atom_oscs, ring_oscs
