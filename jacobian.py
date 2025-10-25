from dataclasses import dataclass
import numpy as np

from config import NUMBER_OF_INTEGRATION_POINTS
from gauss_integration import GAUSS_QUADRATURE

@dataclass
class UniversalJacobian:
  dN_d_epsilon: np.matrix
  dN_d_eta: np.matrix

  def __init__(self):
    self.dN_d_epsilon = np.zeros((NUMBER_OF_INTEGRATION_POINTS**2, 4))
    self.dN_d_eta = np.zeros((NUMBER_OF_INTEGRATION_POINTS**2, 4))

    nodes = np.array(GAUSS_QUADRATURE[NUMBER_OF_INTEGRATION_POINTS]["nodes"])

    for j, eta in enumerate(nodes):
      for i, xi in enumerate(nodes):
          idx = j * NUMBER_OF_INTEGRATION_POINTS + i

          print(i, j, xi, eta, idx)
          self.dN_d_epsilon[idx, 0] = -0.25 * (1 - eta)
          self.dN_d_epsilon[idx, 1] =  0.25 * (1 - eta)
          self.dN_d_epsilon[idx, 2] =  0.25 * (1 + eta)
          self.dN_d_epsilon[idx, 3] = -0.25 * (1 + eta)

          self.dN_d_eta[idx, 0] = -0.25 * (1 - xi)
          self.dN_d_eta[idx, 1] = -0.25 * (1 + xi)
          self.dN_d_eta[idx, 2] =  0.25 * (1 + xi)
          self.dN_d_eta[idx, 3] =  0.25 * (1 - xi)