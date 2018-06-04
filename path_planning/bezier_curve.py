"""
Path Planning with Bezier curve

Original Author: Atsushi Sakai(@Atsushi_twi)
From https://github.com/AtsushiSakai/PythonRobotics
"""
from __future__ import division, print_function

import scipy.special
import numpy as np

import matplotlib.pyplot as plt
import math

show_animation = True

# from constants import MAX_WIDTH, MAX_HEIGHT


def computeControlPoints(x, y, add_current_pos=True):
    """
    The image processing predicts (x, y) points belonging to the line
    :param x: (numpy array)
    :param y: (numpy array)
    :param add_current_pos: (bool)
    :return: (numpy array)
    """
    control_points = np.concatenate((x[None].T, y[None].T), axis=1)

    if add_current_pos:
        current_position = np.array([MAX_WIDTH // 2, MAX_HEIGHT]).reshape(1, -1)
        control_points = np.concatenate((current_position, control_points))

    return control_points


def calcBezierPath(control_points, n_points=100):
    """
    Compute bezier path (trajectory) given control points
    :param control_points: (numpy array)
    :param n_points: (int) n_points in the trajectory
    :return: (numpy array)
    """
    traj = []
    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))

    return np.array(traj)


def bernsteinPoly(n, i, t):
    """
    Bernstein polynom
    :param n: (int) polynom degree
    :param i: (int)
    :param t: (float)
    :return: (float)
    """
    return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)


def bezier(t, control_points):
    """
    Return one point on the bezier curve
    :param t: (float)
    :param control_points: (numpy array)
    :return: (numpy array)
    """
    n = len(control_points) - 1
    return np.sum([bernsteinPoly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)


def coeffDerivative(control_points):
    assert len(control_points) == 4
    P = control_points
    return 3 * np.array([
        P[1] - P[0],
        P[2] - P[1],
        P[3] - P[2]
    ])

def coeffSecondDerivative(control_points):
    assert len(control_points) == 3
    P = control_points
    return 2 * np.array([
        P[1] - P[0],
        P[2] - P[1],
    ])

def bezierIthOrderWeights(control_points, degree):
    # https://pomax.github.io/bezierinfo/#derivatives
    W = {0: control_points}
    for i in range(degree):
        n = len(W[i])
        W[i + 1] = np.array([(n-1) * (W[i][j+1] - W[i][j]) for j in range(n - 1)])
    return W

def curvature(dx, dy, ddx, ddy):
    return (dx*ddy - dy*ddx) / (dx**2 + dy**2)**(3/2)

def tangent(dx, dy):
    d = np.linalg.norm(dx + dy, 2)
    return np.array([dx, dy]) / 2

def calcYaw(dx, dy):
    """
    calc yaw
    """
    return math.atan2(dy, dx)

def calcTrajectory(control_points, n_points=100):
    cp = control_points
    path = calcBezierPath(control_points, 100)
    n_cp = bezierIthOrderWeights(cp, 2)

    rx, ry, ryaw, rk = [], [], [], []
    for t in np.linspace(0, 1, n_points):
        ix, iy = bezier(t, cp)
        dx, dy = bezier(t, n_cp[1])
        ddx, ddy = bezier(t, n_cp[2])
        rx.append(ix)
        ry.append(iy)
        ryaw.append(calcYaw(dx, dy))
        rk.append(curvature(dx, dy, ddx, ddy))

    return rx, ry, ryaw, rk

def calc_4point_bezier_path(sx, sy, syaw, ex, ey, eyaw, offset):
    D = math.sqrt((sx - ex)**2 + (sy - ey)**2) / offset
    cp = np.array(
        [[sx, sy],
         [sx + D * math.cos(syaw), sy + D * math.sin(syaw)],
         [ex - D * math.cos(eyaw), ey - D * math.sin(eyaw)],
         [ex, ey]])

    return cp

def main():
    start_x = 5.0  # [m]
    start_y = 1.0  # [m]
    start_yaw = math.radians(180.0)  # [rad]

    end_x = -6.0  # [m]
    end_y = -3.0  # [m]
    end_yaw = math.radians(-45.0)  # [rad]
    offset = 2

    cp = calc_4point_bezier_path(
        start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset)
    P = calcBezierPath(cp)
    rx, ry, ryaw, rk = calcTrajectory(cp, 100)

    t = 0.8
    n_cp = bezierIthOrderWeights(cp, 2)
    point = bezier(t, cp)
    dt = bezier(t, n_cp[1])
    ddt = bezier(t, n_cp[2])
    cu = curvature(dt[0], dt[1], ddt[0], ddt[1])
    # Normalize derivative
    dt /= np.linalg.norm(dt, 2)
    tangent = np.array([point, point + dt])
    normal = np.array([point, point + [- dt[1], dt[0]]])

    if cu != 0:
        r = 1 / cu
    else:
        r = np.inf

    curvature_center = point + np.array([- dt[1], dt[0]]) * r
    circle = plt.Circle(tuple(curvature_center), r, color=(0, 0.8, 0.8), fill=False, linewidth=1)

    fig, ax = plt.subplots()

    if show_animation:
        ax.plot(rx, ry, label="Bezier Path")
        ax.plot(cp.T[0], cp.T[1], '--o', label="Control Points")
        ax.plot(*bezier(t, cp), '--o', label="Target Point")
        ax.plot(tangent[:, 0], tangent[:, 1], label="Tangent")
        ax.plot(normal[:, 0], normal[:, 1], label="Normal")
        ax.add_artist(circle)
        ax.legend()
        ax.axis("equal")
        ax.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
