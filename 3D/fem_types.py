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
    z: float
    bc_flag: int  # 0 = no BC, 1 = BC applied

@dataclass
class Element:
    node_ids: List[int]
    jacobian: List[Jacobian] = None
    k: float = 0.0
    Q: float = 0.0  # Heat generation per unit volume

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