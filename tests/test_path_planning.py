from __future__ import print_function, division, absolute_import

import subprocess

from .common import *
from constants import *

def testStanleyControlDemo():
    ok = subprocess.call(['python', '-m', 'path_planning.stanley_controller', '--no-display'])
    assertEq(ok, 0)

def testBezierCurveDemo():
    ok = subprocess.call(['python', '-m', 'path_planning.bezier_curve', '--no-display'])
    assertEq(ok, 0)
