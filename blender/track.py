import math
import socket
from collections import namedtuple

import bpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import mathutils

U_MAX_ACC = 25
ANGLE_OFFSET = 0
dt = 0.01
ERROR_MAX = 0.6
HOST = 'localhost'
PORT = 50011

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
        # TODO: add air drag
        self.friction =  friction_coeff * mass
        self.dt = dt
        self.v = 0
        self.acc_norm = 0

    def stepSpeed(self, u_speed):
        sign_speed = np.sign(self.v)
        if self.v == 0:
            self.acc_norm = np.max([0, u_speed - self.friction]) if  u_speed >= 0 else np.min([0, u_speed + self.friction])
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

def compute_bezier_curve(matrix_world, spline):
    # Draw the bezier curve
    # From https://blender.stackexchange.com/questions/688/getting-the-list-of-points-that-describe-a-curve-without-converting-to-mesh
    if len(spline.bezier_points) >= 2:
        r = spline.resolution_u + 1
        segments = len(spline.bezier_points)
        if not spline.use_cyclic_u:
            segments -= 1

        points = []
        for i in range(segments):
            inext = (i + 1) % len(spline.bezier_points)

            knot1 = spline.bezier_points[i].co
            handle1 = spline.bezier_points[i].handle_right
            handle2 = spline.bezier_points[inext].handle_left
            knot2 = spline.bezier_points[inext].co

            _points = mathutils.geometry.interpolate_bezier(knot1, handle1, handle2, knot2, r)
            _points = [matrix_world * vec for vec in _points]
            points.extend(_points)
    return points

def send_to_image_processing(path):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(bytes(path, 'utf-8'))
    data = s.recv(1024)
    path = data.decode("utf-8")
    s.close()
    return np.load(path)

def close_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(bytes("stop", 'utf-8'))
    data = s.recv(1024)
    s.close()

if __name__ == '__main__':
    # Blender objects
    track = bpy.data.objects['track_curve']
    cam = bpy.data.objects['car_camera']
    ANGLE_OFFSET = cam.rotation_euler[2]
    # Render options
    bpy.context.scene.render.resolution_x = 960
    bpy.context.scene.render.resolution_y = 540

    # Init Camera angle/position
    # cam.location[1] = 3.5

    car = Car(Position(cam.location[0], cam.location[1], 0),
              mass=10, friction_coeff=1, dt=dt)

    traj = [[],[], []]

    spline = track.data.splines[0]
    matrix_world = track.matrix_world
    points = compute_bezier_curve(matrix_world, spline)

    u_angle = 0.
    error, errorD, errorI = 0, 0, 0
    last_error = 0

    # Init
    car.pos.x = points[0][0]
    car.pos.y = points[0][1]
    ref_point_idx = 0
    theta_line = 0
    errors = []
    a, b = np.array([0,0]), np.array([0,0])
    turn_percent = 0

    for i in range(500):
        # Write Blender images
        image_path = 'render/{}.png'.format(i)
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)  # render
        # Not in headless mode
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=0)  # hack to draw ui

        p0 = points[ref_point_idx]
        p1 = points[(ref_point_idx + 1) % (len(points) - 1)]

        dist_to_points = [(car.pos.x - p[0])**2 + (car.pos.y - p[1])**2 for p in [p0, p1]]
        ref_point_idx += np.argmin(dist_to_points)
        ref_point_idx %= (len(points) - 1)
        ref_point = points[ref_point_idx]

        # Reference
        # a, b = points[ref_point_idx][:2], points[(ref_point_idx + 1) % len(points)][:2]
        # a, b = np.array(a), np.array(b)

        mat = send_to_image_processing(image_path)
        old_b, old_a, old_turn_percent = a, b, turn_percent
        a, b, infos = mat
        turn_percent, error = infos

        # if error:
        #     a, b, turn_percent = old_a, old_b, old_turn_percent

        h = constrain(turn_percent/100.0, 0, 1)
        v_max = h * 0.2 + (1 - h) * 0.5

        t = constrain(error/float(ERROR_MAX), 0, 1)
        v_min = 0.2
        v = t * v_min + (1 - t) * v_max
        # Constant speed
        car.v = v
        u_speed = 0


        vec = b - a
        # # Dirty fix for good ref angle
        # if abs(vec[0]) < 1e-4:
        #     vec[0] = 1e-4

        # Angle Control
        theta_line = np.arctan2(vec[1], vec[0])
        m = np.array([car.pos.x, car.pos.y])
        dist_to_line = np.linalg.det([b-a, m-a]) / np.linalg.norm(b-a)
        theta_target = theta_line - np.arctan(dist_to_line)

        # errors.append(error)
        # Error between [-pi, pi]
        error = np.arctan(np.tan((theta_target - car.pos.theta)/2))
        if i > 0:
            errorD = error - last_error

        last_error = error

        # PID Control
        u_angle = 0.1 * error + 0.5 * errorD + 0. * errorI
        # u_angle = np.clip(u_angle, -0.005, 0.005)

        # u_angle = 1 * error
        errorI += error

        # Update Car Position
        car.step(u_speed, u_angle, skip_speed=False)

        # Update Blender
        cam.location[0] = car.pos.x
        cam.location[1] = car.pos.y
        cam.rotation_euler[2] = ANGLE_OFFSET + car.pos.theta

        # Trajectory
        traj[0].append(car.pos.x)
        traj[1].append(car.pos.y)
        traj[2].append(convertToDegree(car.pos.theta))

    # print(np.max(errors), np.std(errors), np.mean(errors))
    plt.plot(traj[0], traj[1])
    ax = plt.axes()
    for idx, a in enumerate(traj[2]):
        if idx % 1 == 0:
            # v = 1
            v = car.v
            x,y = traj[0][idx], traj[1][idx]
            ax.arrow(x,y, v*np.cos(convertToRad(a)), v*np.sin(convertToRad(a)), head_width=0.1)

    close_socket()
    plt.show()
