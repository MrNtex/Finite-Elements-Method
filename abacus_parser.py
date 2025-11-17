import re
from dataclasses import dataclass
from typing import List, Dict

from fem_types import GlobalData, Grid, Node, Element

def parse_simulation_file(path: str) -> (GlobalData, Grid):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    global_data = {}
    nodes = []
    elements = []
    bc_nodes = set()

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

        NODE_BUFFER_SIZE = 2
        ELEMENT_BUFFER_SIZE = 4
        if section == "nodes":
            parts = [p.strip() for p in line.split(",") if p.strip()]
            parts = parts[1:]  # Skip node ID
            if len(parts) == NODE_BUFFER_SIZE:
                x, y = parts
                nodes.append(Node(float(x), float(y)))
        elif section == "elements":
            parts = [p.strip() for p in line.split(",") if p.strip()]
            parts = parts[1:]  # Skip element ID
            if len(parts) >= ELEMENT_BUFFER_SIZE:
                node_ids = list(map(int, parts))
                elements.append(Element(node_ids))

        elif section == "bc":
            numbers = [int(x.strip()) for x in line.split(",") if x.strip()]
            bc_nodes.update(numbers)

    global_data_obj = GlobalData(**global_data)
    grid = Grid(nodes, elements, bc_nodes)

    return global_data_obj, grid