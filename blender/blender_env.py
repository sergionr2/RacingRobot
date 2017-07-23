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

U_MAX_ACC = 35
U_MAX_ANGLE = 60  # in rad
ANGLE_OFFSET = 0

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
        self.speed = Speed(0,0)
        self.acc = Acceleration(0,0)
        self.mass = mass
        # TODO: add air friction
        self.friction =  friction_coeff * mass
        self.dt = dt
        self.speed_norm = 0
        self.acc_norm = 0

    def step(self, u_speed, u_angle):
        sign_speed = np.sign(self.speed.x)
        if self.speed.norm() == 0:
            self.acc_norm = np.max([0, u_speed - self.friction]) if  u_speed >= 0 else np.min([0, u_speed + self.friction])
        else:
            self.acc_norm = u_speed - self.friction if sign_speed >= 0 else u_speed + self.friction

        self.acc.update(self.acc_norm, convertToRad(u_angle))
        new_speed = self.speed.x + self.acc.x * self.dt
        if np.sign(self.speed.x) * np.sign(new_speed) == -1:
            self.speed.x = 0
        else:
            self.speed.x = new_speed
            self.speed.y += self.acc.y * self.dt

        self.pos.x += self.speed.x * self.dt
        self.pos.y += self.speed.y * self.dt
        self.pos.theta = u_angle

class PIDControl(object):
    def __init__(self, Kp, Kd, dt, u_max):
        super(PIDControl, self).__init__()
        self.Kp = Kp
        self.Kd = Kd
        self.error = 0
        self.error_derivative = 0
        self.dt = dt
        self.target_pos = 0
        self.u_max = u_max

    def setTarget(self, target_pos):
        self.target_pos = target_pos
        self.error = 0
        self.error_derivative = 0
        self.started = False

    def step(self, current_pos):
        last_error = self.error
        self.error = self.target_pos - current_pos
        if not self.started:
            self.error_derivative = 0
            self.started = True
        else:
            self.error_derivative = self.error - last_error
        u = self.Kp * self.error + (self.Kd / self.dt) * self.error_derivative
        return constrain(u, - self.u_max, self.u_max)



if __name__ == '__main__':
    cam = bpy.data.objects['car_camera']
    # # render options
    bpy.context.scene.render.resolution_x = 960
    bpy.context.scene.render.resolution_y = 540

    # image = bpy.data.images.load(image_path)  # load image
    # image.user_clear()
    # bpy.data.images.remove(image)

    # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=0)  # hack to draw ui
    cam.location[1] = 3.5

    car = Car(Position(cam.location[0], cam.location[1], convertToDegree(cam.rotation_euler[2])),
              mass=10, friction_coeff=2, dt=0.01)

    ANGLE_OFFSET = cam.rotation_euler[2]

    pid_speed = PIDControl(Kp=8, Kd=2, dt=0.01, u_max=U_MAX_ACC)
    pid_speed.setTarget(8)
    pid_angle = PIDControl(Kp=40, Kd=35, dt=0.01, u_max=U_MAX_ANGLE)
    pid_angle.setTarget(4.37)

    hist, hist_v, hist_a, hist_u = [], [], [], []
    traj = [[],[], []]
    for i in range(300):
        u_speed = 25
        # u_speed = pid_speed.step(car.pos.x)
        u_angle = pid_angle.step(car.pos.y)
        car.step(u_speed, u_angle)
        cam.location[0] = car.pos.x
        cam.location[1] = car.pos.y
        cam.rotation_euler[2] = ANGLE_OFFSET + convertToRad(car.pos.theta)
        traj[0].append(car.pos.x)
        traj[1].append(car.pos.y)
        traj[2].append(car.pos.theta)
        # hist.append(car.pos.x)
        # hist_v.append(car.speed.x)
        # hist_a.append(car.acc.x)
        # hist_u.append(u_speed)

        image_path = 'render/{}.png'.format(i)
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)  # render
    # print(pid_angle.error)
    # plt.plot(np.arange(len(hist)), hist, label="x")
    # plt.plot(np.arange(len(hist)), hist_v, label="speed")
    # # plt.plot(np.arange(len(hist)), hist_a, label="acc")
    # # # plt.plot(np.arange(len(hist)), hist_u, label="u")
    # plt.legend()
    plt.plot(traj[0], traj[1])
    ax = plt.axes()
    for idx, a in enumerate(traj[2]):
        if idx % 10 == 0:
            v = 0.1
            x,y = traj[0][idx], traj[1][idx]
            # ax.arrow(x,y, v*np.cos(convertToRad(a)), v*np.sin(convertToRad(a)), head_width=0.1, head_length=v/10, fc='k', ec='k')
            ax.arrow(x,y, v*np.cos(convertToRad(a)), v*np.sin(convertToRad(a)), head_width=0.05)

    plt.show()
