import sys
import os
import glob
import time
import multiprocessing
import numpy as np
from tqdm import tqdm

from fem_types import GlobalData
from mesh_generator.mesh_generator import MeshGeneratorBuilder, PastePattern
from config import MULTIPROCESSING_ENABLED, MAX_PROCESSES, RUN_ALL_PATTERNS
from config_loader import ConfigLoader
from simulate import simulate
from plot_grid import plot_grid 

def run_simulation_task(config_file: str, paste_pattern: PastePattern = None) -> str:
    process_name = multiprocessing.current_process().name
    try:
        cfg = ConfigLoader.load_from_file(config_file)
    except Exception as e:
        return f"[{process_name}] ERROR loading {config_file}: {e}"

    start_time = time.time()

    try:
        generator = (MeshGeneratorBuilder()
                     .set_parameters(cfg.geometry.width, cfg.geometry.depth, cfg.geometry.height)
                     .set_resolution(cfg.geometry.nx, cfg.geometry.ny, cfg.geometry.nz)
                     .set_die_size(cfg.geometry.die_width, cfg.geometry.die_depth)
                     .set_materials(cfg.materials)
                     .set_layers(cfg.layers)
                     .set_power(cfg.power)
                     .set_paste_pattern(paste_pattern if paste_pattern else cfg.paste_pattern)
                     .build())
        
        grid = generator.generate_grid()
    except Exception as e:
        return f"[{process_name}] ERROR generating grid for {config_file}: {e}"

    global_data = GlobalData(
        SimulationTime=cfg.simulation.sim_time,
        SimulationStepTime=cfg.simulation.step_time,
        InitialTemp=cfg.simulation.initial_temp,
        Alpha=cfg.simulation.alpha,
        Tenv=cfg.simulation.ambient_temp,
        WaterTemp=cfg.simulation.water_temp,
        Conductivity=0, Density=0, SpecificHeat=0
    )

    try:
        simulation_history = simulate(grid, global_data)
    except Exception as e:
        return f"[{process_name}] ERROR running simulation {config_file}: {e}"

    duration = time.time() - start_time
    
    final_step = simulation_history[-1]
    max_temp = np.max(final_step)
    output_filename = os.path.splitext(config_file)[0] + "_result.txt"
    with open(output_filename, "w") as f:
        f.write(f"Source Config: {config_file}\n")
        f.write(f"Nodes: {len(grid.nodes)}\n")
        f.write(f"Max Temp Reached: {max_temp:.2f} C\n")
        f.write(f"Compute Time: {duration:.2f} s\n")

    plot_grid(grid, simulation_history)

    return f"[{process_name}] DONE: {os.path.basename(config_file)} -> MaxT: {max_temp:.1f}C ({duration:.1f}s)"

if __name__ == '__main__':
    files_to_run = []
    
    if len(sys.argv) > 1:
        files_to_run = sys.argv[1:]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sim_dir = os.path.join(script_dir, "simulations")
        if os.path.isdir(sim_dir):
            files_to_run = glob.glob(os.path.join(sim_dir, "*.toml"))
        else:
            print(f"Warning: Directory '{sim_dir}' not found. Please create it or pass files as arguments.")

    if not files_to_run:
        print("No input files found.")
        print("Please provide a .toml config file to run the simulation properly.")
        sys.exit(0)

    print(f"--- Found {len(files_to_run)} simulations to run ---")
    for f in files_to_run:
        print(f" - {f}")
    print("-" * 40)

    start_global = time.time()

    if MULTIPROCESSING_ENABLED:
        num_cores = max(1, min(multiprocessing.cpu_count() - 1, MAX_PROCESSES))
        total_tasks = len(files_to_run)
        if RUN_ALL_PATTERNS:
            total_tasks *= len(PastePattern)
        num_workers = min(num_cores, len(files_to_run))    

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = []
            if RUN_ALL_PATTERNS:
                tasks = []
                for file in files_to_run:
                    for pattern in PastePattern:
                        tasks.append((file, pattern))
                for result in tqdm(pool.starmap(run_simulation_task, tasks), total=len(tasks), desc="Simulations Progress"):
                    results.append(result)
            else:
                for result in tqdm(pool.imap_unordered(run_simulation_task, files_to_run), total=len(files_to_run), desc="Simulations Progress"):
                    results.append(result)
    else:
        results = []
        for file in tqdm(files_to_run, desc="Simulations Progress"):
            if RUN_ALL_PATTERNS:
                for pattern in PastePattern:
                    result = run_simulation_task(file, paste_pattern=pattern)
                    results.append(result)
            else:
                result = run_simulation_task(file)
                results.append(result)
    total_time = time.time() - start_global
    
    print("\n--- All Simulations Finished ---")
    print(f"Total batch time: {total_time:.2f} s")
    print("\nResults Summary:")
    for res in sorted(results):
        print(res)