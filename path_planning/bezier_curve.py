"""
Path Planning with Bezier curve

Original Author: Atsushi Sakai(@Atsushi_twi)
From https://github.com/AtsushiSakai/PythonRobotics
"""
from __future__ import division, print_function

import argparse

import scipy.special
import numpy as np
import matplotlib.pyplot as plt

from constants import MAX_WIDTH, MAX_HEIGHT

demo_cp = np.array([[5., 1.], [-2.78, 1.], [-11.5, -4.5], [-6., -8.]])


def computeControlPoints(x, y, add_current_pos=False):
    """
    Compute the control points for creating bezier path
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


def bezierDerivativesControlPoints(control_points, n_derivatives):
    """
    Compute control points of the successive derivatives
    of a given bezier curve
    https://pomax.github.io/bezierinfo/#derivatives
    :param control_points: (numpy array)
    :param n_derivatives: (int)
    ex: n_derivatives=2 -> compute control_points for first and second derivatives
    :return: ([numpy array])
    """
    W = {0: control_points}
    for i in range(n_derivatives):
        n = len(W[i])
        W[i + 1] = np.array([(n - 1) * (W[i][j + 1] - W[i][j]) for j in range(n - 1)])
    return W


def curvature(dx, dy, ddx, ddy):
    """
    Compute curvature at one point
    given first and second derivatives
    :param dx: (float) First derivative along x axis
    :param dy: (float)
    :param ddx: (float) Second derivative along x axis
    :param ddy: (float)
    :return: (float)
    """
    return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)


def calcYaw(dx, dy):
    """
    Compute the yaw angle given the derivative
    at one point on the curve
    :param dx: (float)
    :param dy: (float)
    :return: (float)
    """
    return np.arctan2(dy, dx)


def calcTrajectory(control_points, n_points=100):
    """
    Compute bezier path along with derivative and curvature
    :param control_points: (numpy array)
    :param n_points: (int)
    :return: ([float], [float], [float], [float])
    """
    cp = control_points
    derivatives_cp = bezierDerivativesControlPoints(cp, 2)

    rx, ry, ryaw, rk = [], [], [], []
    for t in np.linspace(0, 1, n_points):
        ix, iy = bezier(t, cp)
        dx, dy = bezier(t, derivatives_cp[1])
        ddx, ddy = bezier(t, derivatives_cp[2])
        rx.append(ix)
        ry.append(iy)
        ryaw.append(calcYaw(dx, dy))
        rk.append(curvature(dx, dy, ddx, ddy))

    return rx, ry, ryaw, rk


def main(show_animation):
    cp = demo_cp
    rx, ry, ryaw, rk = calcTrajectory(cp, 100)

    t = 0.8
    x_target, y_target = bezier(t, cp)
    derivatives_cp = bezierDerivativesControlPoints(cp, 2)
    point = bezier(t, cp)
    dt = bezier(t, derivatives_cp[1])
    ddt = bezier(t, derivatives_cp[2])
    cu = curvature(dt[0], dt[1], ddt[0], ddt[1])
    # Normalize derivative
    dt /= np.linalg.norm(dt, 2)
    tangent = np.array([point, point + dt])
    normal = np.array([point, point + [- dt[1], dt[0]]])
    # Radius of curvature
    r = 1 / cu

    curvature_center = point + np.array([- dt[1], dt[0]]) * r
    circle = plt.Circle(tuple(curvature_center), r, color=(0, 0.8, 0.8), fill=False, linewidth=1)


    if show_animation:  # pragma: no cover
        fig, ax = plt.subplots()
        ax.plot(rx, ry, label="Bezier Path")
        ax.plot(cp.T[0], cp.T[1], '--o', label="Control Points")
        ax.plot(x_target, y_target, '--o', label="Target Point")
        ax.plot(tangent[:, 0], tangent[:, 1], label="Tangent")
        ax.plot(normal[:, 0], normal[:, 1], label="Normal")
        ax.add_artist(circle)
        ax.legend()
        ax.axis("equal")
        ax.grid(True)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a line detector')
    parser.add_argument('--no-display', action='store_true', default=False, help='Do not display plots (for tests)')
    args = parser.parse_args()
    main(not args.no_display)
