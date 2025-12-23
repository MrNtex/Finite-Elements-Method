import numpy as np
from fem_types import Grid, Node, Element
from .mesh_config import MaterialConstants

class MeshGenerator:
    def __init__(self, width, depth, height, nx, ny, nz):
        """
        width, depth, height: Physical dimensions of the 3D object [m]
        nx, ny, nz: Number of elements along each axis
        """
        self.width = width
        self.depth = depth
        self.height = height
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = width / nx
        self.dy = depth / ny
        self.dz = height / nz

    def generate_grid(self, paste_pattern="full"):
        print(f"Generowanie siatki 3D: {self.nx}x{self.ny}x{self.nz} elementów...")
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
                    
                    new_node = Node(node_id_counter, x, y, z)
                    if k == self.nz: 
                        new_node.bc_flag = 1
                    
                    nodes.append(new_node)
                    nodes_map[i, j, k] = node_id_counter
                    node_id_counter += 1

        element_id_counter = 1

        K_SILICON = MaterialConstants.K_SILICON
        K_IHS = MaterialConstants.K_IHS
        K_PASTE = MaterialConstants.K_PASTE
        K_AIR = MaterialConstants.K_AIR
        K_HEATSINK = MaterialConstants.K_HEATSINK

        # 0-1: Silicon, 2-5: IHS, 6: Paste, 7-9: Radiator
        z_silicon_end = int(0.2 * self.nz)
        z_ihs_end = int(0.6 * self.nz)
        z_paste_layer = z_ihs_end

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
                    if k < z_silicon_end:
                        element.k = K_SILICON
                    elif k < z_paste_layer:
                        element.k = K_IHS
                    elif k == z_paste_layer:
                        center_x = (i + 0.5) * self.dx
                        center_y = (j + 0.5) * self.dy
                        
                        if self._is_paste_at(center_x, center_y, paste_pattern):
                            element.k = K_PASTE
                        else:
                            element.k = K_AIR
                    else:
                        element.k = K_HEATSINK
                    
                    elements.append(element)

        print(f"Finished. Generated {len(nodes)} nodes and {len(elements)} elements.")
        
        grid = Grid(nodes, elements)
        return grid

    def _is_paste_at(self, x, y, pattern):
        cx = self.width / 2
        cy = self.depth / 2
        
        if pattern == "full":
            return True
            
        elif pattern == "dot":
            radius = self.width * 0.15
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            return dist <= radius
            
        elif pattern == "x_shape":
            line_width = self.width * 0.1 
            dist1 = abs((x - cx) - (y - cy)) / np.sqrt(2)
            dist2 = abs((x - cx) + (y - cy)) / np.sqrt(2)
            return dist1 < line_width or dist2 < line_width

        return False