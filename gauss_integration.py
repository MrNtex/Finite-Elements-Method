from math import sqrt

GAUSS_QUADRATURE = {
    2: {
        "nodes": [-1/sqrt(3), 1/sqrt(3)],
        "weights": [1, 1]
    },
    3: {
        "nodes": [-sqrt(3/5), 0, sqrt(3/5)],
        "weights": [5/9, 8/9, 5/9]
    },
    4: {
        "nodes": [-0.861136, -0.339981, 0.339981, 0.861136],
        "weights": [0.347855, 0.652145, 0.652145, 0.347855]
    }
}

def gauss_integrate_1d(func, points=2):
    quad = GAUSS_QUADRATURE[points]
    nodes, weights = quad["nodes"], quad["weights"]
    result = 0.0
    for xi, wi in zip(nodes, weights):
        result += wi * func(xi)

    return result

def gauss_integrate_2d(func, points=2):
    quad = GAUSS_QUADRATURE[points]
    nodes, weights = quad["nodes"], quad["weights"]
    result = 0.0
    for xi, wxi in zip(nodes, weights):
        for yi, wyi in zip(nodes, weights):
            result += wxi * wyi * func(xi, yi)
    
    return result