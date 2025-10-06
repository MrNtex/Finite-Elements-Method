import re
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Node:
    id: int
    x: float
    y: float

@dataclass
class Element:
    id: int
    nodes: List[int]

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


def parse_simulation_file(path: str):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    global_data = {}
    nodes = []
    elements = []
    bc_nodes = []

    section = None

    for line in lines:
        if line.startswith("*Node"):
            section = "nodes"
            continue
        elif line.startswith("*Element"):
            section = "elements"
            continue
        elif line.startswith("*BC"):
            section = "bc"
            continue

        if section is None:
            key_value = re.match(r"(\w+)\s+([0-9Ee\.\-]+)", line)
            if key_value:
                key, value = key_value.groups()
                global_data[key] = float(value)
            continue

        NODE_BUFFER_SIZE = 3
        ELEMENT_BUFFER_SIZE = 5
        if section == "nodes":
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) == NODE_BUFFER_SIZE:
                node_id, x, y = parts
                nodes.append(Node(int(node_id), float(x), float(y)))
        elif section == "elements":
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= ELEMENT_BUFFER_SIZE:
                elem_id = int(parts[0])
                node_ids = list(map(int, parts[1:]))
                elements.append(Element(elem_id, node_ids))

        elif section == "bc":
            numbers = [int(x.strip()) for x in line.split(",") if x.strip()]
            bc_nodes.extend(numbers)

    global_data_obj = GlobalData(**global_data)
    grid = Grid(nodes, elements, bc_nodes)

    return global_data_obj, grid


if __name__ == "__main__":
    global_data, grid = parse_simulation_file("mesh.txt")

    print("=== Global Data ===")
    print(global_data)
    print("\n=== Nodes ===")
    for node in grid.nodes:
        print(node)

    print("\n=== Elements ===")
    for elem in grid.elements:
        print(elem)

    print("\n=== Boundary Conditions ===")
    print(grid.bc_nodes)
