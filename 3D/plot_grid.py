from fem_types import Grid, Node, Element

import pyvista as pv
import numpy as np

def plot_grid(grid: Grid, t0: np.ndarray) -> None:
    points = np.array([[n.x, n.y, n.z] for n in grid.nodes])
    cells = []
    cell_type_array = []
    sample_element_len = len(grid.elements[0].node_ids)
    
    if sample_element_len == 8:
        vtk_type = pv.CellType.HEXAHEDRON
    elif sample_element_len == 4:
        vtk_type = pv.CellType.TETRA
    else:
        vtk_type = pv.CellType.HEXAHEDRON 

    for element in grid.elements:
        cells.append(sample_element_len)
        for nid in element.node_ids:
            cells.append(nid - 1)
        
        cell_type_array.append(vtk_type)

    cells = np.array(cells)
    cell_type_array = np.array(cell_type_array)
    mesh = pv.UnstructuredGrid(cells, cell_type_array, points)

    mesh.point_data["Temperature"] = t0
    plotter = pv.Plotter()
    

    plotter.add_mesh(mesh, scalars="Temperature", cmap="jet", show_edges=True)
    plotter.add_axes()
    plotter.add_text(f"Symulacja FEM\nMax Temp: {np.max(t0):.1f} C", position='upper_left')
    plotter.add_mesh_clip_plane(mesh, scalars="Temperature", cmap="jet", assign_to_axis='x')
    plotter.show()