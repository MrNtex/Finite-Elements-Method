from config import DEBUG, SAVE_TO_CSV
from jacobian import UniversalJacobian, calculate_jacobian_for_finite_element
from mesh_generator.mesh_config import get_global_data
from element_matrices import transform_local_derivatives_to_global, calculate_element_matrices
from boundary_matrices import generate_Hbc_matrix_and_P_vector
from mesh_generator.mesh_generator import MeshGenerator

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


if __name__ == '__main__':
    uj = UniversalJacobian()

    global_data = get_global_data()
    generator = MeshGenerator(width=0.04, depth=0.04, height=0.03, nx=15, ny=15, nz=60)

    grid = generator.generate_grid(paste_pattern="full")
    t0 = np.array([global_data.InitialTemp for _ in grid.nodes])
    time = 0

    if SAVE_TO_CSV:
        results_history = {}
        results_history["Time_0.0"] = t0.copy()
        summary_stats = []

    while time < global_data.SimulationTime:
        if DEBUG or True:
            print(f"\n--- Time {time+1}s ---\n")
        aggregated_H_matrix = np.zeros((len(grid.nodes), len(grid.nodes)))
        aggregated_C_matrix = np.zeros((len(grid.nodes), len(grid.nodes)))
        aggregated_P_vector = np.zeros(len(grid.nodes))

        for element in grid.elements:
            element.jacobian = calculate_jacobian_for_finite_element(element, grid, uj)
            if DEBUG:
                print(f"Element {element.node_ids} Jacobians:")
                for j in element.jacobian:
                    print(j)
                    print('\n')

            dN_d_x, dN_d_y, dN_d_z = transform_local_derivatives_to_global(
                uj.dN_d_xi,
                uj.dN_d_eta,
                uj.dN_d_zeta,
                element.jacobian
            )

            if DEBUG:
                print(f"Element {element.node_ids} dN/dx:")
                print(dN_d_x)
                print(f"Element {element.node_ids} dN/dy:")
                print(dN_d_y)

            H_matrix, C_matrix, P_source_vector = calculate_element_matrices(
                dN_d_x,
                dN_d_y,
                dN_d_z,
                uj.N_functions,
                element.jacobian,
                global_data,
                element
            )

            if DEBUG:
                print(f"Element {element.node_ids} H matrix:")
                print(H_matrix)

            if DEBUG:
                print(f"Element {element.node_ids} C matrix:")
                print(C_matrix)

            Hbc_matrix, P_bc_vector = generate_Hbc_matrix_and_P_vector(element, global_data, grid)
            if DEBUG:
                print(f"Element {element.node_ids} Hbc matrix:")
                print(Hbc_matrix)

            H_matrix += Hbc_matrix

            for i_local, node_id_i in enumerate(element.node_ids):
                for j_local, node_id_j in enumerate(element.node_ids):
                    aggregated_H_matrix[node_id_i - 1, node_id_j - 1] += H_matrix[i_local, j_local]
                    aggregated_C_matrix[node_id_i - 1, node_id_j - 1] += C_matrix[i_local, j_local]

            # if DEBUG:
            #         print(f"P Vector: {P_vector} for element {element.node_ids}")

            for i_local, node_id_i in enumerate(element.node_ids):
                aggregated_P_vector[node_id_i - 1] += P_bc_vector[i_local] + P_source_vector[i_local]

        np.set_printoptions(linewidth=np.inf, precision=3, suppress=True)
        if DEBUG:
            print("Aggregated H matrix for the entire grid:")
            print(aggregated_H_matrix)

        if DEBUG:
            print("Aggregated P vector for the entire grid:")
            print(aggregated_P_vector)

        if DEBUG:
            print("Aggregated C matrix for the entire grid:")
            print(aggregated_C_matrix)

        eq_left_matrix = aggregated_H_matrix + (aggregated_C_matrix / global_data.SimulationStepTime)
        if DEBUG:
            print("Equation left matrix:")
            print(eq_left_matrix)
        eq_right_matrix = aggregated_P_vector + ((aggregated_C_matrix / global_data.SimulationStepTime) @ t0)
        if DEBUG:
            print("Equation right matrix:")
            print(eq_right_matrix)
        t0 = np.linalg.solve(eq_left_matrix, eq_right_matrix)
        if DEBUG or True:
            print("Resulting t matrix for the entire grid:")
            #print(t0)
            print("Min: ", np.min(t0), " Max: ", np.max(t0))

        time += global_data.SimulationStepTime
        if SAVE_TO_CSV:
            results_history[f"Time_{time}"] = t0.copy()
            summary_stats.append({
                "Time": time,
                "MinTemp": round(np.min(t0), 2),
                "MaxTemp": round(np.max(t0), 2),
            })

    if SAVE_TO_CSV:
        print("\nSaving results to CSV file...")
        df_summary = pd.DataFrame(summary_stats)
        filename_summary = "simulation_min_max.csv"
        df_summary.to_csv(filename_summary, index=False)
        print(f"Saved min/max summary to {filename_summary}")