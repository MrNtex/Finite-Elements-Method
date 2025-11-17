from config import DEBUG
from gauss_integration import gauss_integrate_1d, gauss_integrate_2d
from jacobian import UniversalJacobian, calculate_jacobian_for_finite_element
from abacus_parser import parse_simulation_file
from H_matrix import transform_local_derivatives_to_global, generate_H_matrix
from Hbc_matrix import generate_Hbc_matrix

import numpy as np

if __name__ == '__main__':
    uj = UniversalJacobian()

    global_data, grid = parse_simulation_file("Test2_4_4_MixGrid.txt")

    aggregated_H_matrix = np.zeros((len(grid.nodes), len(grid.nodes)))

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
        
        H_matrix = generate_H_matrix(
            dN_d_x,
            dN_d_y,
            element.jacobian,
            global_data
        )
        if DEBUG:
            print(f"Element {element.node_ids} H matrix:")
            print(H_matrix)

        Hbc_matrix = generate_Hbc_matrix(element, global_data, grid)

        if DEBUG or True:
            print(f"Element {element.node_ids} Hbc matrix:")
            print(Hbc_matrix)
        for i_local, node_id_i in enumerate(element.node_ids):
            for j_local, node_id_j in enumerate(element.node_ids):
                #print(f"Adding H[{i_local},{j_local}] ({H_matrix[i_local, j_local]}) to global H[{node_id_i - 1},{node_id_j - 1}] ({aggregated_H_matrix[node_id_i - 1, node_id_j - 1]})")
                aggregated_H_matrix[node_id_i - 1, node_id_j - 1] += H_matrix[i_local, j_local]

    if DEBUG or True:
        np.set_printoptions(linewidth=np.inf, precision=3, suppress=True)
        print("Aggregated H matrix for the entire grid:")
        print(aggregated_H_matrix)