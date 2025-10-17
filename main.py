from gauss_integration import gauss_integrate_1d, gauss_integrate_2d

print("test")
if __name__ == '__main__':
    f1d = lambda x: 5*x**2 + 3*x + 6
    f2d = lambda x,y: 5*x**2*y**2 + 3*x*y + 6

    a, b = 0, 1
    ax, bx = 0, 1
    ay, by = 0, 1

    res_1d_2pt = gauss_integrate_1d(f1d, a, b, points=3)
    res_1d_3pt = gauss_integrate_1d(f1d, a, b, points=4)
    res_2d_2pt = gauss_integrate_2d(f2d, ax, bx, ay, by, points=2)
    res_2d_3pt = gauss_integrate_2d(f2d, ax, bx, ay, by, points=3)

    print("1D Gauss 2-point:", res_1d_2pt)
    print("1D Gauss 3-point:", res_1d_3pt)
    print("2D Gauss 2-point:", res_2d_2pt)
    print("2D Gauss 3-point:", res_2d_3pt)