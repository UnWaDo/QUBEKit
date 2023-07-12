import math
import numpy as np
from typing import List, Tuple, Union

from qubekit.engines.multiwfn import CritPoints


def select_points_by_type(
    cps: Tuple[List[CritPoints], List[np.ndarray]],
    selector: List[CritPoints]
) -> List[np.ndarray]:
    indices = [i for i, t in enumerate(cps[0]) if t in selector]
    return [cps[1][i] for i in indices]


def calc_cosine(vector1: np.ndarray, vector2: np.ndarray) -> float:
    return np.inner(vector1, vector2) / np.linalg.norm(vector1) / np.linalg.norm(vector2)


def cosine_metrics(
    vector1: np.ndarray,
    vector2: np.ndarray,
    target: float
) -> float:
    return np.abs(calc_cosine(vector1, vector2) - target)

def angle_cosine_metrics(
    vector1: np.ndarray,
    vector2: np.ndarray,
    target: float
) -> float:
    return np.abs(
        np.abs(np.arccos(calc_cosine(vector1, vector2))) - target
    )

def select_two_on_120(
    atom_coords: np.ndarray,
    neighbours_coords: List[np.ndarray],
    cps: List[np.ndarray]
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple]:
    neighbour_v = neighbours_coords[0] - atom_coords
    neighbour_v /= np.linalg.norm(neighbour_v)
    selected = [cps[0], cps[1]]
    selected_metrics = list(map(lambda x:
        cosine_metrics(x - atom_coords, neighbour_v, -0.5),
        selected
    ))
    if selected_metrics[0] > selected_metrics[1]:
        selected = [selected[1], selected[0]]
        selected_metrics = [selected_metrics[1], selected_metrics[0]]
    for cp in cps[2:]:
        cp_metrics = cosine_metrics(cp - atom_coords, neighbour_v, -0.5)
        if cp_metrics < selected_metrics[1]:
            selected[1] = cp
            selected_metrics[1] = cp_metrics
            if selected_metrics[0] > selected_metrics[1]:
                selected = [selected[1], selected[0]]
                selected_metrics = [selected_metrics[1], selected_metrics[0]]
    if sum([metric <= 0.4 for metric in selected_metrics]) != 2:
        return tuple()
    return tuple(selected)


def select_one_on_reverse(
    atom_coords: np.ndarray,
    neighbours_coords: List[np.ndarray],
    cps: List[np.ndarray]
) -> Union[Tuple[np.ndarray], Tuple]:
    def metric(x):
        cos = cosine_metrics(
            x - atom_coords,
            median,
            -1
        ) / 0.01
        return (cos ** 2 + cos + 1) / np.linalg.norm(x - atom_coords)
    median = np.zeros(3)
    for coords in neighbours_coords:
        vector = coords - atom_coords
        median += vector / np.linalg.norm(vector)
    median /= np.linalg.norm(median)
    selected = cps[0]
    selected_metrics = metric(selected)
    for cp in cps[1:]:
        cp_metrics = metric(cp)
        if cp_metrics < selected_metrics:
            selected = cp
            selected_metrics = cp_metrics
    if not selected_metrics <= 10:
        return tuple()
    return tuple([selected])


def select_two_tetrahedral(
    atom_coords: np.ndarray,
    neighbours_coords: List[np.ndarray],
    cps: List[np.ndarray]
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple]:
    def metric(x):
        return angle_cosine_metrics(
            np.cross(
                vector,
                x - atom_coords
            ),
            plane,
            np.pi * 2 / 3
        ) + np.abs(
            np.linalg.norm(
                x - ref_points[0]
            ) - np.linalg.norm(
                x - ref_points[1]
            )
        ) 
    plane = np.cross(
        neighbours_coords[1] - neighbours_coords[0],
        atom_coords - neighbours_coords[0]
    )
    plane /= np.linalg.norm(plane)
    vector = neighbours_coords[0] - atom_coords
    vector /= np.linalg.norm(vector)
    ref_points = np.array([neighbours_coords[0] - atom_coords, neighbours_coords[1] - atom_coords])
    ref_points /= np.linalg.norm(ref_points, axis=1).reshape(-1, 1)
    ref_points += atom_coords
    selected = [cps[0], cps[1]]
    selected_metrics = list(map(metric, selected))
    if selected_metrics[0] > selected_metrics[1]:
        selected = [selected[1], selected[0]]
        selected_metrics = [selected_metrics[1], selected_metrics[0]]
    for cp in cps[2:]:
        cp_metrics = metric(cp)
        if cp_metrics < selected_metrics[1]:
            selected[1] = cp
            selected_metrics[1] = cp_metrics
            if selected_metrics[0] > selected_metrics[1]:
                selected = [selected[1], selected[0]]
                selected_metrics = [selected_metrics[1], selected_metrics[0]]
    if sum([metric <= 0.8 for metric in selected_metrics]) != 2:
        return tuple()
    return tuple(selected)


def select_one_at_center(
    center_coords: np.ndarray,
    cps: List[np.ndarray]
) -> Union[Tuple[np.ndarray], Tuple]:
    selected = cps[0]
    selected_metrics = np.linalg.norm(selected - center_coords)
    for cp in cps[1:]:
        cp_metrics = np.linalg.norm(cp - center_coords)
        if cp_metrics < selected_metrics:
            selected = cp
            selected_metrics = cp_metrics
    if not selected_metrics <= 0.1:
        return tuple()
    return tuple([selected])


def are_coplanar(coordinates: List[np.ndarray]) -> bool:
    plane = np.cross(
        coordinates[0] - coordinates[1],
        coordinates[2] - coordinates[1]
    )
    plane /= np.linalg.norm(plane)
    vector = coordinates[2] - coordinates[1]
    for coordinate in coordinates[3:]:
        new_plane = np.cross(
            coordinate - coordinates[1],
            vector
        )
        new_plane /= np.linalg.norm(new_plane)
        metric = abs(np.dot(plane, new_plane))
        if not math.isclose(metric, 1, abs_tol=1e-2):
            return False
    return True


def are_colinear(coordinates: List[np.ndarray]) -> bool:
    line = coordinates[1] - coordinates[0]
    line /= np.linalg.norm(line)
    for coordinate in coordinates[2:]:
        vector = coordinate - coordinates[0]
        vector /= np.linalg.norm(vector)
        metric = abs(np.dot(line, vector))
        if not math.isclose(metric, 1, abs_tol=1e-4):
            return False
    return True
