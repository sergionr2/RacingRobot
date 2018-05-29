"""
Path Planning with 4 point Beizer curve

Author: Atsushi Sakai(@Atsushi_twi)
From https://github.com/AtsushiSakai/PythonRobotics
"""
import scipy.special
import numpy as np


def calcBezierPath(control_points, n_points=4):
    traj = []
    for t in np.linspace(0, 1, 100):
        traj.append(bezier(n_points - 1, t, control_points))

    return np.array(traj)


def bernsteinPoly(n, i, t):
    """
    Bernstein polynom
    """
    return scipy.special.comb(n, i) * t**i * (1 - t)**(n - i)


def bezier(n, t, q):
    p = np.zeros(2)
    for i in range(n + 1):
        p += bernsteinPoly(n, i, t) * q[i]
    return p
