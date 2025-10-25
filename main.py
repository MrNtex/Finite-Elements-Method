from gauss_integration import gauss_integrate_1d, gauss_integrate_2d
from jacobian import UniversalJacobian

print("test")
if __name__ == '__main__':
    uj = UniversalJacobian()
    print(uj.dN_d_epsilon)
    print(uj.dN_d_eta)