from fem_types import Grid
import pyvista as pv
import numpy as np
from typing import List

def plot_grid(grid: Grid, results_history: List[np.ndarray]) -> None:
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

    global_min = min(np.min(step) for step in results_history)
    global_max = max(np.max(step) for step in results_history)
    mesh.point_data["Temperature"] = results_history[0]
    
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh, 
        scalars="Temperature", 
        cmap="jet", 
        show_edges=True, 
        clim=[global_min, global_max]
    )
    
    plotter.add_axes()
    plotter.add_text(f"FEM Simulation\nMax Temp: {global_max:.1f} C", position='upper_left')
    def update_step(value):
        index = int(value)
        if 0 <= index < len(results_history):
            mesh.point_data["Temperature"] = results_history[index]

    plotter.add_slider_widget(
        update_step,
        [0, len(results_history) - 1],
        title="Time Step",
        value=0,
        fmt="%0.f"
    )
    plotter.add_mesh_clip_plane(
        mesh, 
        scalars="Temperature", 
        cmap="jet", 
        assign_to_axis='x',
        clim=[global_min, global_max]
    )

    plotter.show()