from dataclasses import dataclass
import numpy as np

from config import NUMBER_OF_INTEGRATION_POINTS
from gauss_integration import GAUSS_QUADRATURE

@dataclass
class UniversalJacobian:
  dN_d_epsilon: np.matrix
  dN_d_eta: np.matrix

def get_universal_jacobian() -> UniversalJacobian:
    dN_d_xi = np.zeros((NUMBER_OF_INTEGRATION_POINTS**2, 4))
    dN_d_eta = np.zeros((NUMBER_OF_INTEGRATION_POINTS**2, 4))

    nodes = np.array(GAUSS_QUADRATURE[NUMBER_OF_INTEGRATION_POINTS]["nodes"])

    for j, eta in enumerate(nodes):
      for i, xi in enumerate(nodes):
          idx = j * NUMBER_OF_INTEGRATION_POINTS + i

          print(i, j, xi, eta, idx)
          dN_d_xi[idx, 0] = -0.25 * (1 - eta)
          dN_d_xi[idx, 1] =  0.25 * (1 - eta)
          dN_d_xi[idx, 2] =  0.25 * (1 + eta)
          dN_d_xi[idx, 3] = -0.25 * (1 + eta)

          dN_d_eta[idx, 0] = -0.25 * (1 - xi)
          dN_d_eta[idx, 1] = -0.25 * (1 + xi)
          dN_d_eta[idx, 2] =  0.25 * (1 + xi)
          dN_d_eta[idx, 3] =  0.25 * (1 - xi)

    return UniversalJacobian(dN_d_epsilon=dN_d_xi, dN_d_eta=dN_d_eta)
