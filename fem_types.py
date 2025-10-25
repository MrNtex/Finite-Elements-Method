import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Jacobian:
    J: np.matrix
    invJ: np.matrix
    detJ: float

    def __repr__(self):
        return f"Jacobian(detJ={self.detJ}, J=\n{self.J}, invJ=\n{self.invJ})"

@dataclass
class Node:
    x: float
    y: float

@dataclass
class Element:
    node_ids: List[int]
    jacobian: List[Jacobian] = None

@dataclass
class GlobalData:
    SimulationTime: float
    SimulationStepTime: float
    Conductivity: float
    Alfa: float
    Tot: float
    InitialTemp: float
    Density: float
    SpecificHeat: float

@dataclass
class Grid:
    nodes: List[Node]
    elements: List[Element]
    bc_nodes: List[int]