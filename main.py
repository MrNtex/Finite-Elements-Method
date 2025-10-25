from gauss_integration import gauss_integrate_1d, gauss_integrate_2d
from jacobian import UniversalJacobian, calculate_jacobian_for_finite_element
from abacus_parser import parse_simulation_file

print("test")
if __name__ == '__main__':
    uj = UniversalJacobian()
    
    global_data, grid = parse_simulation_file("Test2_4_4_MixGrid.txt")

    for element in grid.elements:
        element.jacobian = calculate_jacobian_for_finite_element(element, grid, uj)
        print(f"Element {element.node_ids} Jacobians:")
        for j in element.jacobian:
            print(j)
            print('\n')