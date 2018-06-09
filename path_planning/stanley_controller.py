"""

Path tracking simulation with Stanley steering control and PID speed control.

Original Author: Atsushi Sakai(@Atsushi_twi)
From https://github.com/AtsushiSakai/PythonRobotics
"""
from __future__ import division, print_function

import sys

import numpy as np
import matplotlib.pyplot as plt

from .bezier_curve import calcTrajectory, demo_cp


k = 1  # control gain
Kp = 1.0  # speed propotional gain
dt = 0.05  # [s] time difference
L = 1.5  # [m] Wheel base of vehicle
max_steer = np.radians(20.0)  # [rad] max steering angle

show_animation = True


class State(object):
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def updateState(state, acceleration, delta):
    delta = np.clip(delta, -max_steer, max_steer)

    state.x += state.v * np.cos(state.yaw) * dt
    state.y += state.v * np.sin(state.yaw) * dt
    state.yaw += state.v / L * np.tan(delta) * dt
    state.yaw = normalizeAngle(state.yaw)
    state.v += acceleration * dt

    return state


def speedControl(target, current):
    a = Kp * (target - current)
    return a


def stanleyControl(state, cx, cy, cyaw, pind):

    ind, efa = calcTargetIndex(state, cx, cy)

    if pind >= ind:
        ind = pind

    theta_e = normalizeAngle(cyaw[ind] - state.yaw)
    theta_d = np.arctan2(k * efa, state.v)
    delta = theta_e + theta_d

    return delta, ind


def normalizeAngle(angle):
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calcTargetIndex(state, cx, cy):

    # calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = [np.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]
    mind = min(d)
    ind = d.index(mind)

    tyaw = normalizeAngle(np.arctan2(fy - cy[ind], fx - cx[ind]) - state.yaw)
    if tyaw > 0.0:
        mind = - mind

    return ind, mind


def main():
    cp = demo_cp
    cx, cy, cyaw, ck = calcTrajectory(cp, 200)

    target_speed = 30.0 / 3.6  # [m/s]

    T = 100.0  # max simulation time

    # initial state
    state = State(x=5, y=1.0, yaw=np.radians(-180.0), v=0)

    lastIndex = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_ind, mind = calcTargetIndex(state, cx, cy)

    while T >= time and lastIndex > target_ind:
        # Compute Acceleration
        ai = speedControl(target_speed, state.v)
        di, target_ind = stanleyControl(state, cx, cy, cyaw, target_ind)
        state = updateState(state, ai, di)

        time = time + dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if show_animation:
            plt.cla()
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert lastIndex >= target_ind, "Cannot goal"

    if show_animation:
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        flg, ax = plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
