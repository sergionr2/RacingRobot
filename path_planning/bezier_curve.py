"""
Path Planning with Bezier curve

Original Author: Atsushi Sakai(@Atsushi_twi)
From https://github.com/AtsushiSakai/PythonRobotics
"""
import scipy.special
import numpy as np

from constants import MAX_WIDTH, MAX_HEIGHT


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
