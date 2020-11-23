import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from RosslerSolver import simulator

def trajectory(X, Y, Z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(X, Y, Z, 'red', alpha=0.5)
    ax.set_xlabel('X', fontsize=15, fontweight='bold')
    ax.set_ylabel('Y', fontsize=15, fontweight='bold')
    ax.set_zlabel('Z', fontsize=15, fontweight='bold')
    fig.savefig('plots/3D.jpg')

def project_xy(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X, Y, 'red', alpha=0.5)
    ax.set_xlabel('X', fontsize=15, fontweight='bold')
    ax.set_ylabel('Y', fontsize=15, fontweight='bold')
    fig.savefig('plots/xy.jpg')

def project_yz(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Y, Z, 'red', alpha=0.5)
    ax.set_xlabel('Y', fontsize=15, fontweight='bold')
    ax.set_ylabel('Z', fontsize=15, fontweight='bold')
    fig.savefig('plots/yz.jpg')

def project_xz(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X, Z, 'red', alpha=0.5)
    ax.set_xlabel('X', fontsize=15, fontweight='bold')
    ax.set_ylabel('Z', fontsize=15, fontweight='bold')
    fig.savefig('plots/xz.jpg')

def time_x(X):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = [i for i in range(len(X))]
    ax.plot(t, X, 'red', alpha=0.5)
    ax.set_xlabel('time', fontsize=15, fontweight='bold')
    ax.set_ylabel('X', fontsize=15, fontweight='bold')
    fig.savefig('plots/time_x.jpg')

def time_y(Y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = [i for i in range(len(X))]
    ax.plot(t, Y, 'red', alpha=0.5)
    ax.set_xlabel('time', fontsize=15, fontweight='bold')
    ax.set_ylabel('Y', fontsize=15, fontweight='bold')
    fig.savefig('plots/time_y.jpg')

def time_z(Z):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = [i for i in range(len(X))]
    ax.plot(t, Z, 'red', alpha=0.5)
    ax.set_xlabel('time', fontsize=15, fontweight='bold')
    ax.set_ylabel('Z', fontsize=15, fontweight='bold')
    fig.savefig('plots/time_z.jpg')

def butterfly_effect():
    a, b, c = 0.2, 0.2, 5.7
    x0, y0, z0 = 1, 2, 3
    x1, y1, z1 = 1.01, 2, 3
    X0, Y0, Z0 = simulator(x0, y0, z0, a, b, c, 0.01, 20000)
    X1, Y1, Z1 = simulator(x1, y1, z1, a, b, c, 0.01, 20000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = [i for i in range(len(X))]
    ax.plot(t, X0, 'red', alpha=0.5, label="initial x={} y={} z={}".format(x0, y0, z0))
    ax.plot(t, X1, 'blue', alpha=0.5, label="initial x={} y={} z={}".format(x1, y1, z1))
    ax.set_xlabel('time', fontsize=15, fontweight='bold')
    ax.set_ylabel('X', fontsize=15, fontweight='bold')
    ax.legend()
    fig.savefig('plots/butterfly_x.jpg')

if __name__ == "__main__":
    a, b, c = 0.2, 0.2, 5.7
    x0, y0, z0 = 1, 2, 3
    X, Y, Z = simulator(x0, y0, z0, a, b, c, 0.01, 20000)

    trajectory(X, Y, Z)
    project_xy(X, Y, Z)
    project_xz(X, Y, Z)
    project_yz(X, Y, Z)
    time_x(X)
    time_y(Y)
    time_z(Z)
    butterfly_effect()
