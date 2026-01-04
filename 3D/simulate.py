from typing import List
from config import DEBUG, SAVE_TO_CSV, PLOT_SAVE_INTERVAL
from jacobian import UniversalJacobian, calculate_jacobian_for_finite_element
from mesh_generator.mesh_config import get_global_data
from element_matrices import transform_local_derivatives_to_global, calculate_element_matrices
from boundary_matrices import generate_Hbc_matrix_and_P_vector
from fem_types import Grid, GlobalData

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

def simulate(grid: Grid, global_data: GlobalData) -> List[np.ndarray]:
    uj = UniversalJacobian()
    t0 = np.array([global_data.InitialTemp for _ in grid.nodes])
    current_time = 0

    print(f"Nodes: {len(grid.nodes)}")
    print("--- STARTING ASSEMBLY OF MATRICES (This may take a moment...) ---")

    global_H = lil_matrix((len(grid.nodes), len(grid.nodes)))
    global_C = lil_matrix((len(grid.nodes), len(grid.nodes)))
    global_P = np.zeros(len(grid.nodes))

    for i, element in tqdm(enumerate(grid.elements), total=len(grid.elements)):
        element.jacobian = calculate_jacobian_for_finite_element(element, grid, uj)
        dN_d_x, dN_d_y, dN_d_z = transform_local_derivatives_to_global(
            uj.dN_d_xi, uj.dN_d_eta, uj.dN_d_zeta, element.jacobian
        )

        H_local, C_local, P_source_local = calculate_element_matrices(
            dN_d_x, dN_d_y, dN_d_z, uj.N_functions, element.jacobian, global_data, element
        )

        Hbc_local, P_bc_local = generate_Hbc_matrix_and_P_vector(element, global_data, grid)
        H_local += Hbc_local
        
        for i_loc, node_id_i in enumerate(element.node_ids):
            idx_i = node_id_i - 1
            global_P[idx_i] += P_source_local[i_loc] + P_bc_local[i_loc]

            for j_loc, node_id_j in enumerate(element.node_ids):
                idx_j = node_id_j - 1
                
                global_H[idx_i, idx_j] += H_local[i_loc, j_loc]
                global_C[idx_i, idx_j] += C_local[i_loc, j_loc]

    print("--- END OF ASSEMBLY. CONVERTING TO CSR... ---")
    global_H = global_H.tocsr()
    global_C = global_C.tocsr()
    
    dt = global_data.SimulationStepTime
    lhs_matrix = global_H + (global_C / dt)

    dirichlet_indices = [i for i, node in enumerate(grid.nodes) if node.dirichlet_bc]
    lhs_matrix = lhs_matrix.tolil()

    for idx in dirichlet_indices:
        lhs_matrix[idx, :] = 0
        lhs_matrix[idx, idx] = 1.0
    lhs_matrix = lhs_matrix.tocsr()
    
    if SAVE_TO_CSV:
        results_history = {}
        results_history["Time_0.0"] = t0.copy()
        summary_stats = []

    print("--- STARTING TIME SIMULATION ---")
    plot_update_interval = PLOT_SAVE_INTERVAL
    last_plot_time = -plot_update_interval

    simulation_history = [t0.copy()]
    
    while current_time < global_data.SimulationTime:
        rhs_vector = global_P + (global_C.dot(t0) / dt)
        for idx in dirichlet_indices:
            rhs_vector[idx] = global_data.WaterTemp
        
        t0 = spsolve(lhs_matrix, rhs_vector)
        
        current_time += dt
        min_t = np.min(t0)
        max_t = np.max(t0)

        if current_time - last_plot_time >= plot_update_interval:
            simulation_history.append(t0.copy())
            last_plot_time = current_time
        
        if DEBUG or True:
            print(f"Time: {current_time:.2f}s | Min: {min_t:.2f} | Max: {max_t:.2f}")

        if SAVE_TO_CSV:
            summary_stats.append({
                "Time": current_time,
                "MinTemp": round(min_t, 2),
                "MaxTemp": round(max_t, 2),
            })

    if SAVE_TO_CSV:
        print("\nSaving results...")
        df_summary = pd.DataFrame(summary_stats)
        df_summary.to_csv("simulation_optimized.csv", index=False)
        print("Saved.")

    return simulation_history