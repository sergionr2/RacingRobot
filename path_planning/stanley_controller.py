"""

Path tracking simulation with Stanley steering control and PID speed control.
https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf

Original Author: Atsushi Sakai(@Atsushi_twi)
From https://github.com/AtsushiSakai/PythonRobotics
"""
from __future__ import division, print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np

from .bezier_curve import calcTrajectory, demo_cp
from constants import K_STANLEY_CONTROL, CAR_LENGTH, MAX_STEERING_ANGLE,\
                    MAX_SPEED_STRAIGHT_LINE, MIN_SPEED, MIN_RADIUS, MAX_RADIUS

Kp_speed = 5  # speed propotional gain
dt = 0.05  # [s] time difference


class State(object):
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        """
        :param acceleration: (float) Speed control
        :param delta: (float) Steering control
        """
        delta = np.clip(delta, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / CAR_LENGTH * np.tan(delta) * dt
        self.yaw = normalizeAngle(self.yaw)
        self.v += acceleration * dt


def stanleyControl(state, cx, cy, cyaw, last_target_idx):
    """
    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :param cyaw: [float]
    :param last_target_idx: (int)
    :return: (float, float, float)
    """
    # Cross track error
    current_target_idx, error_front_axle = calcTargetIndex(state, cx, cy)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # theta_e corrects the heading error
    theta_e = normalizeAngle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(K_STANLEY_CONTROL * error_front_axle, state.v)
    # Steering control
    delta = theta_e + theta_d

    return delta, current_target_idx, error_front_axle


def normalizeAngle(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calcTargetIndex(state, cx, cy):
    """
    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + CAR_LENGTH * np.cos(state.yaw)
    fy = state.y + CAR_LENGTH * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = [np.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]
    error_front_axle = min(d)
    target_idx = d.index(error_front_axle)

    target_yaw = normalizeAngle(np.arctan2(fy - cy[target_idx], fx - cx[target_idx]) - state.yaw)
    if target_yaw > 0.0:
        error_front_axle = - error_front_axle

    return target_idx, error_front_axle


def main(show_animation):
    cp = demo_cp
    cx, cy, cyaw, ck = calcTrajectory(cp, n_points=200)

    target_speed = MAX_SPEED_STRAIGHT_LINE / 3.6  # [m/s]
    max_simulation_time = 100.0

    # initial state
    state = State(x=10, y=5.0, yaw=np.radians(-180.0), v=0)

    last_idx = len(cx) - 1
    current_t = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    cross_track_errors = []
    curvatures = []
    target_idx, _ = calcTargetIndex(state, cx, cy)
    max_speed = target_speed
    max_radius = 40
    min_radius = 5

    while max_simulation_time >= current_t and last_idx > target_idx:
        # Compute Acceleration
        acceleration = Kp_speed * (target_speed - state.v)
        delta, target_idx, cross_track_error = stanleyControl(state, cx, cy, cyaw, target_idx)
        state.update(acceleration, delta)
        cross_track_errors.append(cross_track_error)
        if ck[target_idx] > 0:
            current_radius = 1 / ck[target_idx]
        else:
            current_radius = np.inf

        h = 1 - (np.clip(current_radius, MIN_RADIUS, MAX_RADIUS) - MIN_RADIUS) / (MAX_RADIUS - MIN_RADIUS)
        target_speed = h * MIN_SPEED / 3.6 + (1 - h) * MAX_SPEED_STRAIGHT_LINE / 3.6

        curvatures.append(current_radius)

        current_t += dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(current_t)

        if show_animation:  # pragma: no cover
            plt.cla()
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.arrow(state.x, state.y, np.cos(state.yaw), np.sin(state.yaw),
                      fc="r", ec="k", head_width=0.5, head_length=0.5)
            plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    if show_animation:  # pragma: no cover
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        _, ax1 = plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)

        fig, ax2 = plt.subplots(1)
        plt.plot(t[1:], cross_track_errors, "-r", label="cross_track_error")
        plt.plot(t[1:], curvatures, "-b", label="curvature radius")
        plt.xlabel("Time[s]")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a line detector')
    parser.add_argument('--no-display', action='store_true', default=False, help='Do not display plots (for tests)')
    args = parser.parse_args()
    main(not args.no_display)
