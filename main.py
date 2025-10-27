from gauss_integration import gauss_integrate_1d, gauss_integrate_2d
from jacobian import UniversalJacobian, calculate_jacobian_for_finite_element
from abacus_parser import parse_simulation_file
from H_matrix import transform_local_derivatives_to_global, generate_H_matrix

if __name__ == '__main__':
    uj = UniversalJacobian()
    
    global_data, grid = parse_simulation_file("Test2_4_4_MixGrid.txt")

    for element in grid.elements:
        element.jacobian = calculate_jacobian_for_finite_element(element, grid, uj)
        print(f"Element {element.node_ids} Jacobians:")
        for j in element.jacobian:
            print(j)
            print('\n')

        dN_d_x, dN_d_y = transform_local_derivatives_to_global(
            uj.dN_d_epsilon,
            uj.dN_d_eta,
            element.jacobian
        )

        print(f"Element {element.node_ids} dN/dx:")
        print(dN_d_x)
        print(f"Element {element.node_ids} dN/dy:")
        print(dN_d_y)
        
        H_matrix = generate_H_matrix(
            dN_d_x,
            dN_d_y,
            element.jacobian
        )
        print(f"Element {element.node_ids} H matrix:")
        print(H_matrix)