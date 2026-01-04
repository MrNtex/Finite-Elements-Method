from mesh_generator.mesh_config import get_global_data
from mesh_generator.mesh_generator import MeshGeneratorBuilder, PastePattern
from simulate import simulate
from plot_grid import plot_grid
from units import Distance

if __name__ == '__main__':
    global_data = get_global_data()
    
    generator = MeshGeneratorBuilder().set_parameters(
        width=Distance.cm(3),
        depth=Distance.cm(3),
        height=Distance.cm(2)
    ).set_resolution(
        nx=25,
        ny=25,
        nz=30
    ).set_paste_pattern(PastePattern.X_SHAPE
    ).build()
    grid = generator.generate_grid()
    
    simulation_history = simulate(grid, global_data)
    plot_grid(grid, simulation_history)