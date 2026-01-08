from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from enum import Enum
from fem_types import Grid, Node, Element
from units import Distance
from config import DEBUG, ENTIRE_RADIATOR_HAS_DERICHLET_BC


class PastePattern(Enum):
    FULL = "full"
    DOT = "dot"
    X_SHAPE = "x_shape"
    TWO_LINES = "two_lines"


@dataclass
class MaterialProperties:
    k: float
    rho: float
    cp: float


@dataclass
class MaterialConfig:
    silicon: MaterialProperties
    ihs: MaterialProperties
    paste: MaterialProperties
    heatsink: MaterialProperties
    air: MaterialProperties
    substrate: MaterialProperties


@dataclass
class LayerConfig:
    silicon: float
    ihs: float
    paste: float


@dataclass
class GeometryParameters:
    width: float = Distance.cm(4)
    depth: float = Distance.cm(4)
    height: float = Distance.cm(3)
    nx: int = 25
    ny: int = 25
    nz: int = 30
    die_width: float = Distance.mm(15)
    die_depth: float = Distance.mm(12)


class MeshGeneratorBuilder:
    def __init__(self):
        self._geometry = GeometryParameters()
        self._materials: MaterialConfig | None = None
        self._layers: LayerConfig | None = None
        self._power: float = 95.0
        self._pattern: PastePattern = PastePattern.FULL

    def set_parameters(
        self, width: float, depth: float, height: float
    ) -> MeshGeneratorBuilder:
        self._geometry.width = width
        self._geometry.depth = depth
        self._geometry.height = height
        return self

    def set_resolution(self, nx: int, ny: int, nz: int) -> MeshGeneratorBuilder:
        self._geometry.nx = nx
        self._geometry.ny = ny
        self._geometry.nz = nz
        return self

    def set_die_size(self, die_width: float, die_depth: float) -> MeshGeneratorBuilder:
        self._geometry.die_width = die_width
        self._geometry.die_depth = die_depth
        return self

    def set_materials(self, materials: MaterialConfig) -> MeshGeneratorBuilder:
        self._materials = materials
        return self

    def set_layers(self, layers: LayerConfig) -> MeshGeneratorBuilder:
        self._layers = layers
        return self

    def set_power(self, power: float) -> MeshGeneratorBuilder:
        self._power = power
        return self

    def set_paste_pattern(self, paste_pattern: PastePattern) -> MeshGeneratorBuilder:
        self._pattern = paste_pattern
        return self

    def build(self) -> MeshGenerator:
        if (
            self._geometry.width < self._geometry.die_width
            or self._geometry.depth < self._geometry.die_depth
        ):
            raise ValueError("Error: Die dimensions exceed overall mesh dimensions.")
        if self._materials is None:
            raise ValueError("Error: Material configuration not set.")
        if self._layers is None:
            raise ValueError("Error: Layer configuration not set.")

        return MeshGenerator(
            self._geometry, self._materials, self._layers, self._power, self._pattern
        )


class MeshGenerator:
    def __init__(
        self,
        geometry: GeometryParameters,
        materials: MaterialConfig,
        layers: LayerConfig,
        power: float,
        pattern: PastePattern,
    ):

        self.geo = geometry
        self.materials = materials
        self.layers = layers
        self.power = power
        self.pattern = pattern

        self.dx = self.geo.width / self.geo.nx
        self.dy = self.geo.depth / self.geo.ny
        self.dz = self.geo.height / self.geo.nz

    def generate_grid(self) -> Grid:
        print(
            f"Generating 3D mesh: {self.geo.nx}x{self.geo.ny}x{self.geo.nz} elements..."
        )

        n_silicon = int(self.geo.nz * (self.layers.silicon / 100))
        n_ihs = int(self.geo.nz * (self.layers.ihs / 100))
        n_paste = int(self.geo.nz * (self.layers.paste / 100))

        n_silicon = max(1, n_silicon)
        n_ihs = max(1, n_ihs)
        n_paste = max(1, n_paste)

        n_heatsink = self.geo.nz - n_silicon - n_ihs - n_paste

        idx_silicon_end = n_silicon
        idx_ihs_end = idx_silicon_end + n_ihs
        idx_paste_end = idx_ihs_end + n_paste
        idx_radiator_start = idx_paste_end

        print(f"Layer distribution (k indices):")
        print(f" - Silicon:  0  to {idx_silicon_end - 1} \t({n_silicon} layers)")
        print(f" - IHS:      {idx_silicon_end} to {idx_ihs_end - 1} \t({n_ihs} layers)")
        print(
            f" - Paste:    {idx_ihs_end} to {idx_paste_end - 1} \t({n_paste} layers) -> Pattern: {self.pattern}"
        )
        print(
            f" - Heatsink: {idx_paste_end} to {self.geo.nz - 1} \t({n_heatsink} layers)"
        )

        if n_heatsink < 0:
            raise ValueError(
                "Error: Layer configuration results in negative heatsink layers. Increase nz."
            )

        die_volume = self.geo.die_width * self.geo.die_depth * n_silicon * self.dz
        silicon_Q = self.power / die_volume
        print(f"Heat Source Power: {self.power} W")
        print(f"Heat Source Density (Q): {silicon_Q/1e6:.2f} MW/m^3")

        nodes = []
        elements = []
        node_id_counter = 1
        nodes_map = np.zeros(
            (self.geo.nx + 1, self.geo.ny + 1, self.geo.nz + 1), dtype=int
        )

        for k in range(self.geo.nz + 1):
            for j in range(self.geo.ny + 1):
                for i in range(self.geo.nx + 1):
                    x = i * self.dx
                    y = j * self.dy
                    z = k * self.dz

                    new_node = Node(x, y, z)

                    is_side_x = i == 0 or i == self.geo.nx
                    is_side_y = j == 0 or j == self.geo.ny
                    is_top = k == self.geo.nz
                    in_radiator_zone = k >= idx_radiator_start

                    new_node.dirichlet_bc = False
                    new_node.convection_bc = False

                    if is_top or (
                        in_radiator_zone and ENTIRE_RADIATOR_HAS_DERICHLET_BC
                    ):
                        new_node.dirichlet_bc = True
                    elif is_side_x or is_side_y:
                        new_node.convection_bc = True

                    nodes.append(new_node)
                    nodes_map[i, j, k] = node_id_counter
                    node_id_counter += 1

        for k in range(self.geo.nz):
            for j in range(self.geo.ny):
                for i in range(self.geo.nx):
                    n_ids = [
                        nodes_map[i, j, k],
                        nodes_map[i + 1, j, k],
                        nodes_map[i + 1, j + 1, k],
                        nodes_map[i, j + 1, k],
                        nodes_map[i, j, k + 1],
                        nodes_map[i + 1, j, k + 1],
                        nodes_map[i + 1, j + 1, k + 1],
                        nodes_map[i, j + 1, k + 1],
                    ]
                    element = Element(n_ids)
                    center_x = (i + 0.5) * self.dx
                    center_y = (j + 0.5) * self.dy

                    if k < idx_silicon_end:
                        if self._is_inside_die(center_x, center_y):
                            element.Q = silicon_Q
                            element.k = self.materials.silicon.k
                            element.rho = self.materials.silicon.rho
                            element.cp = self.materials.silicon.cp
                            if (
                                DEBUG
                                and k == 0
                                and i == self.geo.nx // 2
                                and j == self.geo.ny // 2
                            ):
                                print(f"DEBUG: Center element is Silicon Source")
                        else:
                            element.k = self.materials.substrate.k
                            element.rho = self.materials.substrate.rho
                            element.cp = self.materials.substrate.cp

                    elif k < idx_ihs_end:
                        element.k = self.materials.ihs.k
                        element.rho = self.materials.ihs.rho
                        element.cp = self.materials.ihs.cp

                    elif k < idx_paste_end:
                        if self._is_paste_at(center_x, center_y, self.pattern):
                            element.k = self.materials.paste.k
                            element.rho = self.materials.paste.rho
                            element.cp = self.materials.paste.cp
                        else:
                            element.k = self.materials.air.k
                            element.rho = self.materials.air.rho
                            element.cp = self.materials.air.cp

                    else:
                        element.k = self.materials.heatsink.k
                        element.rho = self.materials.heatsink.rho
                        element.cp = self.materials.heatsink.cp

                    elements.append(element)

        print(f"Finished. Generated {len(nodes)} nodes and {len(elements)} elements.")
        return Grid(nodes, elements)

    def _is_inside_die(self, x: float, y: float) -> bool:
        cx = self.geo.width / 2
        cy = self.geo.depth / 2
        half_w = self.geo.die_width / 2
        half_d = self.geo.die_depth / 2
        return (cx - half_w) <= x <= (cx + half_w) and (cy - half_d) <= y <= (
            cy + half_d
        )

    def _is_paste_at(self, x: float, y: float, pattern: PastePattern) -> bool:
        cx = self.geo.width / 2
        cy = self.geo.depth / 2

        if pattern == PastePattern.FULL:
            return True
        elif pattern == PastePattern.DOT:
            radius = min(self.geo.die_width, self.geo.die_depth) * 0.8
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            return dist <= radius
        elif pattern == PastePattern.X_SHAPE:
            line_width = self.geo.die_width * 0.15
            dist1 = abs((x - cx) - (y - cy)) / np.sqrt(2)
            dist2 = abs((x - cx) + (y - cy)) / np.sqrt(2)
            return dist1 < line_width or dist2 < line_width

        elif pattern == PastePattern.TWO_LINES:
            line_thickness = self.geo.die_width * 0.1
            pos1 = cx - (self.geo.die_width * 0.25)
            pos2 = cx + (self.geo.die_width * 0.25)
            is_line1 = abs(x - pos1) < (line_thickness / 2)
            is_line2 = abs(x - pos2) < (line_thickness / 2)
            return is_line1 or is_line2

        return False
