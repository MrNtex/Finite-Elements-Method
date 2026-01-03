from __future__ import annotations
from fem_types import Grid, Node, Element
from .mesh_config import MaterialConstants, MaterialHeights, CPU_POWER

import numpy as np
from dataclasses import dataclass
from enum import Enum

class PastePattern(Enum):
    FULL = "full"
    DOT = "dot"
    X_SHAPE = "x_shape"
    TWO_LINES = "two_lines"

@dataclass
class MeshParameters:
    width: float = 0.04
    depth: float = 0.04
    height: float = 0.03
    nx: int = 25
    ny: int = 25
    nz: int = 30

    die_width: float = 0.015
    die_depth: float = 0.012

@dataclass
class MeshGeneratorBuilder:
    def __init__(self):
        self._config = MeshParameters()
    
    def set_parameters(self, width: float, depth: float, height: float) -> MeshGeneratorBuilder:
        self._config.width = width
        self._config.depth = depth
        self._config.height = height
        return self

    def set_resolution(self, nx: int, ny: int, nz: int) -> MeshGeneratorBuilder:
        self._config.nx = nx
        self._config.ny = ny
        self._config.nz = nz
        return self
    
    def set_die_size(self, die_width: float, die_depth: float) -> MeshGeneratorBuilder:
        self._config.die_width = die_width
        self._config.die_depth = die_depth
        return self
    
    def build(self) -> MeshGenerator:
        if self._config.width < self._config.die_width or self._config.depth < self._config.die_depth:
            raise ValueError("Error: Die dimensions exceed overall mesh dimensions.")
        return MeshGenerator(self._config)

class MeshGenerator:
    def __init__(self, params: MeshParameters):
        """
        width, depth, height: Physical dimensions of the 3D object [m]
        nx, ny, nz: Number of elements along each axis
        """
        self.width = params.width
        self.depth = params.depth
        self.height = params.height
        self.nx = params.nx
        self.ny = params.ny
        self.nz = params.nz

        self.dx = self.width / self.nx
        self.dy = self.depth / self.ny
        self.dz = self.height / self.nz

        self.die_width = params.die_width
        self.die_depth = params.die_depth

    def generate_grid(self, paste_pattern: PastePattern = PastePattern.FULL) -> Grid:
        print(f"Generating 3D mesh: {self.nx}x{self.ny}x{self.nz} elements...")
        n_silicon = int(self.nz * (MaterialHeights.SILICON_HEIGHT / 100))
        n_ihs     = int(self.nz * (MaterialHeights.IHS_HEIGHT / 100))
        
        n_paste   = int(self.nz * (MaterialHeights.PASTE_HEIGHT / 100))
        if n_paste < 1 or n_silicon < 1 or n_ihs < 1:
            raise ValueError("Error: Material layer heights too small for the given nz. Increase nz or layer heights.")
        n_heatsink = self.nz - n_silicon - n_ihs - n_paste
        idx_silicon_end = n_silicon
        idx_ihs_end     = idx_silicon_end + n_ihs
        idx_paste_end   = idx_ihs_end + n_paste

        print(f"Layer distribution (k indices):")
        print(f" - Silicon:  0  to {idx_silicon_end - 1} \t({n_silicon} layers)")
        print(f" - IHS:      {idx_silicon_end} to {idx_ihs_end - 1} \t({n_ihs} layers)")
        print(f" - Paste:    {idx_ihs_end} to {idx_paste_end - 1} \t({n_paste} layers) -> Pattern: {paste_pattern}")
        print(f" - Heatsink: {idx_paste_end} to {self.nz - 1} \t({n_heatsink} layers)")

        if n_heatsink < 0:
            raise ValueError("Error: Layer configuration results in negative heatsink layers. Adjust material heights or total nz.")

        die_volume = (self.die_width *
                      self.die_depth *
                      n_silicon * self.dz)
        silicon_Q = CPU_POWER / die_volume  # W/m^3
        print(f"Heat Source Density (Q): {silicon_Q} W/m^3")

        nodes = []
        elements = []

        node_id_counter = 1
        nodes_map = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=int)

        for k in range(self.nz + 1):      #  Z (height)
            for j in range(self.ny + 1):  # Y (depth)
                for i in range(self.nx + 1): # X (width)
                    x = i * self.dx
                    y = j * self.dy
                    z = k * self.dz
                    
                    new_node = Node(x, y, z)

                    is_side_x = (i == 0 or i == self.nx)
                    is_side_y = (j == 0 or j == self.ny)
                    in_radiator_zone = (k >= idx_paste_end)

                    if in_radiator_zone:
                        new_node.dirichlet_bc = True
                        new_node.convection_bc = False
                    elif (is_side_x or is_side_y) and in_radiator_zone:
                        new_node.dirichlet_bc = False
                        new_node.convection_bc = True

                    nodes.append(new_node)
                    nodes_map[i, j, k] = node_id_counter
                    node_id_counter += 1

        MC = MaterialConstants

        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    n_ids = [
                        nodes_map[i,   j,   k],   # 0
                        nodes_map[i+1, j,   k],   # 1
                        nodes_map[i+1, j+1, k],   # 2
                        nodes_map[i,   j+1, k],   # 3
                        nodes_map[i,   j,   k+1], # 4
                        nodes_map[i+1, j,   k+1], # 5
                        nodes_map[i+1, j+1, k+1], # 6
                        nodes_map[i,   j+1, k+1]  # 7
                    ]
                    element = Element(n_ids)
                    center_x = (i + 0.5) * self.dx
                    center_y = (j + 0.5) * self.dy
                    if k < idx_silicon_end:
                        element.k = MC.K_SILICON
                        element.rho = MC.RHO_SILICON
                        element.cp = MC.C_SILICON
                        if self._is_inside_die(center_x, center_y):
                            print(f"Assigning heat source to element at layer {k}, position ({center_x:.4f}, {center_y:.4f})")
                            element.Q = silicon_Q
                    elif k < idx_ihs_end:
                        element.k = MC.K_IHS
                        element.rho = MC.RHO_IHS
                        element.cp = MC.C_IHS
                    elif k < idx_paste_end:
                        center_x = (i + 0.5) * self.dx
                        center_y = (j + 0.5) * self.dy
                        
                        if self._is_paste_at(center_x, center_y, paste_pattern):
                            element.k = MC.K_PASTE
                            element.rho = MC.RHO_PASTE
                            element.cp = MC.C_PASTE
                        else:
                            element.k = MC.K_AIR
                            element.rho = MC.RHO_AIR
                            element.cp = MC.C_AIR
                    else:
                        element.k = MC.K_HEATSINK
                        element.rho = MC.RHO_HEATSINK
                        element.cp = MC.C_HEATSINK

                    elements.append(element)

        print(f"Finished. Generated {len(nodes)} nodes and {len(elements)} elements.")
        
        grid = Grid(nodes, elements)
        return grid
    
    def _is_inside_die(self, x: float, y: float) -> bool:
        cx = self.width / 2
        cy = self.depth / 2
        half_w = self.die_width / 2
        half_d = self.die_depth / 2

        return (cx - half_w) <= x <= (cx + half_w) and (cy - half_d) <= y <= (cy + half_d)

    def _is_paste_at(self, x: float, y: float, pattern: PastePattern) -> bool:
        cx = self.width / 2
        cy = self.depth / 2

        if pattern == PastePattern.FULL:
            return True

        elif pattern == PastePattern.DOT:
            radius = self.width * 0.15
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            return dist <= radius

        elif pattern == PastePattern.X_SHAPE:
            line_width = self.width * 0.1 
            dist1 = abs((x - cx) - (y - cy)) / np.sqrt(2)
            dist2 = abs((x - cx) + (y - cy)) / np.sqrt(2)
            return dist1 < line_width or dist2 < line_width
        elif pattern == PastePattern.TWO_LINES:
            line_thickness = self.width * 0.1
            pos1 = self.width * 0.33
            pos2 = self.width * 0.66
            
            is_line1 = abs(x - pos1) < (line_thickness / 2)
            is_line2 = abs(x - pos2) < (line_thickness / 2)
            return is_line1 or is_line2

        return False