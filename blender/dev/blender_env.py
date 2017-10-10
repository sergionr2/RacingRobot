import math
from collections import namedtuple

import bpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn


class Position(object):
    def __init__(self, x, y, theta=0):
        super(Position, self).__init__()
        self.x, self.y = x, y
        self.theta = theta

    def norm(self):
        return np.linalg.norm([self.x, self.y])

    def update(self, norm, theta):
        self.x = norm * np.cos(theta)
        self.y = norm * np.sin(theta)


class Speed(Position):
    def __init__(self, vx, vy):
        super(Speed, self).__init__(vx, vy)


class Acceleration(Position):
    def __init__(self, ax, ay):
        super(Acceleration, self).__init__(ax, ay)


U_MAX_ACC = 25
ANGLE_OFFSET = 0
dt = 0.01


def convertToDegree(angle):
    return (angle * 180) / np.pi


def convertToRad(angle):
    return (angle * np.pi) / 180


def constrain(x, a, b):
    return np.max([a, np.min([x, b])])


class Car(object):
    def __init__(self, start_pos, mass, friction_coeff=1, dt=0.01):
        super(Car, self).__init__()
        self.pos = start_pos
        self.speed = Speed(0, 0)
        self.acc = Acceleration(0, 0)
        self.mass = mass
        # TODO: add air drag
        self.friction = friction_coeff * mass
        self.dt = dt
        self.v = 0
        self.acc_norm = 0

    def stepSpeed(self, u_speed):
        sign_speed = np.sign(self.v)
        if self.v == 0:
            self.acc_norm = np.max([0, u_speed - self.friction]) if u_speed >= 0 else np.min(
                [0, u_speed + self.friction])
        else:
            self.acc_norm = u_speed - self.friction if sign_speed >= 0 else u_speed + self.friction
        new_speed = self.v + self.acc_norm * self.dt
        if np.sign(self.v) * np.sign(new_speed) == -1:
            self.v = 0
        else:
            self.v = new_speed

    def step(self, u_speed, u_angle, skip_speed=False):
        if not skip_speed:
            self.stepSpeed(u_speed)
        theta = car.pos.theta
        car.pos.x += self.v * np.cos(theta)
        car.pos.y += self.v * np.sin(theta)
        car.pos.theta += u_angle


class PIDControl(object):
    def __init__(self, Kp, Kd, dt, u_max):
        super(PIDControl, self).__init__()
        self.Kp = Kp
        self.Kd = Kd
        self.error = 0
        self.error_derivative = 0
        self.dt = dt
        self.u_max = u_max

    def step(self, error, t=0):
        last_error = self.error
        self.error = error
        if t > 0:
            self.error_derivative = self.error - last_error

        u = self.Kp * self.error + (self.Kd / self.dt) * self.error_derivative
        return constrain(u, - self.u_max, self.u_max)


if __name__ == '__main__':
    cam = bpy.data.objects['car_camera']
    # Render options
    bpy.context.scene.render.resolution_x = 960
    bpy.context.scene.render.resolution_y = 540

    # image = bpy.data.images.load(image_path)  # load image
    # image.user_clear()
    # bpy.data.images.remove(image)

    # Init Camera
    cam.location[1] = 3.5
    ANGLE_OFFSET = cam.rotation_euler[2]

    car = Car(Position(cam.location[0], cam.location[1], 0),
              mass=10, friction_coeff=2, dt=dt)

    pid_speed = PIDControl(Kp=2, Kd=2, dt=dt, u_max=U_MAX_ACC)
    x_target = 7
    y_target = 4.35
    theta_line = 0

    hist, hist_v, hist_a, hist_u = [], [], [], []
    traj = [[], [], []]
    u_angle = 0.
    error, errorD = 0, 0
    last_error = 0
    for i in range(160):
        error_x = x_target - car.pos.x
        u_speed = pid_speed.step(error_x, i)

        # Limit Speed
        if car.v > 0.1:
            u_speed = np.sign(u_speed) * car.friction

        # Constant speed
        # car.v = 0.1
        # u_speed = 0

        # Angle Control
        dist_to_line = car.pos.y - y_target
        theta_target = theta_line - np.arctan(dist_to_line)

        last_error = error
        # Error between [-pi, pi]
        error = np.arctan(np.tan((theta_target - car.pos.theta) / 2))
        if i > 0:
            errorD = error - last_error

        # PD Control
        u_angle = 1 * error + 0.0 * (errorD / dt)

        # Update Car Position
        car.step(u_speed, u_angle, skip_speed=False)
        # Update Blender
        cam.location[0] = car.pos.x
        cam.location[1] = car.pos.y
        cam.rotation_euler[2] = ANGLE_OFFSET + car.pos.theta

        # Trajectory
        traj[0].append(car.pos.x)
        traj[1].append(car.pos.y)
        traj[2].append(car.pos.theta)

        # Position, Speed, Acceleration
        # hist.append(car.pos.x)
        # hist_v.append(car.speed.x)
        # hist_a.append(car.acc.x)
        # hist_u.append(u_speed)


        # Write Blender images
        image_path = 'render/{}.png'.format(i)
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)  # render

    # plt.plot(np.arange(len(hist)), hist, label="x")
    # plt.plot(np.arange(len(hist)), hist_v, label="speed")
    # plt.plot(np.arange(len(hist)), hist_a, label="acc")
    # plt.plot(np.arange(len(hist)), hist_u, label="u")
    # plt.legend()

    plt.plot(traj[0], traj[1])
    ax = plt.axes()
    for idx, a in enumerate(traj[2]):
        if idx % 10 == 0:
            v = 0.01
            x, y = traj[0][idx], traj[1][idx]
            ax.arrow(x, y, v * np.cos(convertToRad(a)), v * np.sin(convertToRad(a)), head_width=0.05)

    plt.show()
