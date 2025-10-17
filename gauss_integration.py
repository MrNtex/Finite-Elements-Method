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
        "weights": [0.347855, 0.652145, 0.347855, 0.652145]
    }
}

def gauss_integrate_1d(func, a, b, points=2):
    quad = GAUSS_QUADRATURE[points]
    nodes, weights = quad["nodes"], quad["weights"]
    result = 0.0
    for xi, wi in zip(nodes, weights):
        x_mapped = (b - a)/2 * xi + (a + b)/2
        result += wi * func(x_mapped)
    result *= (b - a)/2
    return result

def gauss_integrate_2d(func, ax, bx, ay, by, points=2):
    quad = GAUSS_QUADRATURE[points]
    nodes, weights = quad["nodes"], quad["weights"]
    result = 0.0
    for i, xi in enumerate(nodes):
        for j, yj in enumerate(nodes):
            x_mapped = (bx - ax)/2 * xi + (ax + bx)/2
            y_mapped = (by - ay)/2 * yj + (ay + by)/2
            result += weights[i] * weights[j] * func(x_mapped, y_mapped)
    result *= (bx - ax)/2 * (by - ay)/2
    return result