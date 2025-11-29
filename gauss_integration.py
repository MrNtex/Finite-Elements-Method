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
        "nodes": [
            -sqrt(3/7 + (2/7)*sqrt(6/5)), 
            -sqrt(3/7 - (2/7)*sqrt(6/5)), 
            sqrt(3/7 - (2/7)*sqrt(6/5)), 
            sqrt(3/7 + (2/7)*sqrt(6/5))
        ],
        "weights": [
            (18 - sqrt(30))/36, 
            (18 + sqrt(30))/36, 
            (18 + sqrt(30))/36, 
            (18 - sqrt(30))/36
        ]
    },
    5: {
        "nodes": [
            -(1/3)*sqrt(5 + 2*sqrt(10/7)), 
            -(1/3)*sqrt(5 - 2*sqrt(10/7)), 
            0, 
            (1/3)*sqrt(5 - 2*sqrt(10/7)), 
            (1/3)*sqrt(5 + 2*sqrt(10/7))
        ],
        "weights": [
            (322 - 13*sqrt(70))/900, 
            (322 + 13*sqrt(70))/900, 
            128/225, 
            (322 + 13*sqrt(70))/900, 
            (322 - 13*sqrt(70))/900
        ]
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