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
