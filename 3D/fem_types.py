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
    convection_bc: bool = False
    dirichlet_bc: bool = False

@dataclass
class Element:
    node_ids: List[int]
    jacobian: List[Jacobian] = None
    k: float = 0.0
    rho: float = 0.0
    cp: float = 0.0
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
    WaterTemp: float

@dataclass
class Grid:
    nodes: List[Node]
    elements: List[Element]