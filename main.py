from config import DEBUG
from gauss_integration import gauss_integrate_1d, gauss_integrate_2d
from jacobian import UniversalJacobian, calculate_jacobian_for_finite_element
from abacus_parser import parse_simulation_file
from Element_matrices import transform_local_derivatives_to_global, generate_H_and_C_matrix
from Boudary_matrices import generate_Hbc_matrix_and_P_vector

import numpy as np

if __name__ == '__main__':
    uj = UniversalJacobian()

    global_data, grid = parse_simulation_file("Test1_4_4.txt")
    t0 = np.array([global_data.InitialTemp for _ in grid.nodes])
    time = 0

    while time < global_data.SimulationTime:
        if DEBUG or True:
            print(f"\n--- Time step {time} ---\n")
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

            dN_d_x, dN_d_y = transform_local_derivatives_to_global(
                uj.dN_d_epsilon,
                uj.dN_d_eta,
                element.jacobian
            )

            if DEBUG:
                print(f"Element {element.node_ids} dN/dx:")
                print(dN_d_x)
                print(f"Element {element.node_ids} dN/dy:")
                print(dN_d_y)

            H_matrix, C_matrix = generate_H_and_C_matrix(
                dN_d_x,
                dN_d_y,
                uj.N_functions,
                element.jacobian,
                global_data
            )

            if DEBUG:
                print(f"Element {element.node_ids} H matrix:")
                print(H_matrix)

            if DEBUG:
                print(f"Element {element.node_ids} C matrix:")
                print(C_matrix)

            Hbc_matrix, P_vector = generate_Hbc_matrix_and_P_vector(element, global_data, grid)
            if DEBUG:
                print(f"Element {element.node_ids} Hbc matrix:")
                print(Hbc_matrix)

            H_matrix += Hbc_matrix

            for i_local, node_id_i in enumerate(element.node_ids):
                for j_local, node_id_j in enumerate(element.node_ids):
                    aggregated_H_matrix[node_id_i - 1, node_id_j - 1] += H_matrix[i_local, j_local]
                    aggregated_C_matrix[node_id_i - 1, node_id_j - 1] += C_matrix[i_local, j_local]

            if DEBUG:
                    print(f"P Vector: {P_vector} for element {element.node_ids}")

            for i_local, node_id_i in enumerate(element.node_ids):
                aggregated_P_vector[node_id_i - 1] += P_vector[i_local]

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
        print("Equation left matrix:")
        print(eq_left_matrix)
        eq_right_matrix = aggregated_P_vector + ((aggregated_C_matrix / global_data.SimulationStepTime) @ t0)
        print("Equation right matrix:")
        print(eq_right_matrix)
        t0 = np.linalg.solve(eq_left_matrix, eq_right_matrix)
        if DEBUG or True:
            print("Resulting t matrix for the entire grid:")
            #print(t0)
            print("Min: ", np.min(t0), " Max: ", np.max(t0))
        time += global_data.SimulationStepTime