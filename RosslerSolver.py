"""
this script solve Rossler's system given initial condition
"""

def x_dot(x, y, z, a, b, c):
    return - y - z

def y_dot(x, y, z, a, b, c):
    return x + a * y

def z_dot(x, y, z, a, b, c):
    return b + z * (x - c)

def RK4(x, y, z, a, b, c, h):
    """
    :param x: current x
    :param y: current y
    :param z: current z
    :param h: step h
    :return: (next_step_x, next_step_y, next_step_z)
    """
    x_next, y_next, z_next = x, y, z

    # slope at t_0
    k1x = x_dot(x, y, z, a, b, c)
    k1y = y_dot(x, y, z, a, b, c)
    k1z = z_dot(x, y, z, a, b, c)

    # slope at t_0 + h / 2
    xx = x + h * k1x / 2
    yy = y + h * k1y / 2
    zz = z + h * k1z / 2
    k2x = x_dot(xx, yy, zz, a, b, c)
    k2y = y_dot(xx, yy, zz, a, b, c)
    k2z = z_dot(xx, yy, zz, a, b, c)

    # second slope at t_0 + h / 2
    xx = x + h * k2x / 2
    yy = y + h * k2y / 2
    zz = z + h * k2z / 2
    k3x = x_dot(xx, yy, zz, a, b, c)
    k3y = y_dot(xx, yy, zz, a, b, c)
    k3z = z_dot(xx, yy, zz, a, b, c)

    # slope at t_0 + h
    xx = x + h * k3x
    yy = y + h * k3y
    zz = z + h * k3z
    k4x = x_dot(xx, yy, zz, a, b, c)
    k4y = y_dot(xx, yy, zz, a, b, c)
    k4z = z_dot(xx, yy, zz, a, b, c)

    x_next = x + 1.0 / 6.0 * h * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_next = y + 1.0 / 6.0 * h * (k1y + 2 * k2y + 2 * k3y + k4y)
    z_next = z + 1.0 / 6.0 * h * (k1z + 2 * k2z + 2 * k3z + k4z)

    return x_next, y_next, z_next


def simulator(x0, y0, z0, a, b, c, h, N):
    X, Y, Z = [x0], [y0], [z0]
    for _ in range(N):
        x, y, z = RK4(X[-1], Y[-1], Z[-1], a, b, c, h)
        X.append(x)
        Y.append(y)
        Z.append(z)
    return X, Y, Z


if __name__ == "__main__":
    a, b, c = 0.2, 0.2, 5.7
    x0, y0, z0 = 1, 2, 3
    X, Y, Z = simulator(x0, y0, z0, a, b, c, 0.01, 1000)
    print(X, Y, Z)




