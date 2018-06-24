from __future__ import print_function, division, absolute_import

import subprocess

import pytest
import numpy as np

from .common import *
from constants import *
from path_planning.bezier_curve import demo_cp, calcBezierPath
from path_planning.stanley_controller import normalizeAngle

def testStanleyControlDemo():
    ok = subprocess.call(['python', '-m', 'path_planning.stanley_controller', '--no-display'])
    assertEq(ok, 0)

def testBezierCurveDemo():
    ok = subprocess.call(['python', '-m', 'path_planning.bezier_curve', '--no-display'])
    assertEq(ok, 0)


def testCalcBezierPath():
    path = calcBezierPath(demo_cp, n_points=10)
    assertEq(path[0, 0], demo_cp[0, 0])
    assertEq(path[0, 1], demo_cp[0, 1])

    assertEq(path[-1, 0], demo_cp[-1, 0])
    assertEq(path[-1, 1], demo_cp[-1, 1])


def testNormalizeAngle():
    angle = np.pi / 3 + 4 * (2 * np.pi)
    assertEq(normalizeAngle(angle), pytest.approx(np.pi / 3))

    angle = np.pi / 4 - 5 * (2 * np.pi)
    assertEq(normalizeAngle(angle), pytest.approx(np.pi / 4))
