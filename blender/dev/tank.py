import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn

def f(x, u):
    theta = x[2]
    u = np.clip(u, -0.1, 0.1)
    return np.array([np.cos(theta), np.sin(theta), u])

def custom_range(start=0, end=0, step=1):
    i = start
    while i < end:
        yield i
        i += step

a = np.array([-30, -4])
b = np.array([30, 30])
x = np.array([-25, -10, 0], dtype=np.float64)

fig = plt.figure("fig")
line = plt.Line2D([a[0], b[0]], [a[1], b[1]], lw=2.5)
plt.gca().add_line(line)

hist = []
total_time = 50
error = 0
errorD = 0
last_error = None
Kp, Kd = 1, 30
dt = 0.2
theta_line = np.arctan2(b[1]-a[1], b[0]-a[0])

for t in custom_range(0, total_time, dt):
    m = x[0:2]
    dist_to_line = np.linalg.det([b-a, m-a]) / np.linalg.norm(b-a)
    theta_target = theta_line - np.arctan(dist_to_line)

    error = np.arctan(np.tan((theta_target - x[2])/2))

    if last_error is None:
        errorD = 0
        last_error = error
    else:
        errorD = error - last_error
        last_error = error

    u = Kp * error + Kd * errorD
    x += f(x, u) * dt
    hist.append(x.copy())


plt.axes().set_xlim([-30, 30])
plt.axes().set_ylim([-30, 30])
ax = plt.axes()

def convertToDegree(angle):
    return (angle * 180) / np.pi

def convertToRad(angle):
    return (angle * np.pi) / 180


def draw_triangle(x, y, theta):
    w = 2
    l = 8
    # theta = convertToRad(theta)
    pt1 = x - w * np.sin(theta), y + w * np.cos(theta)
    pt2 = x + l * np.cos(theta), y + l * np.sin(theta)
    pt3 = x + w * np.sin(theta), y - w * np.cos(theta)
    return [pt1, pt2, pt3]

patch = plt.Polygon(draw_triangle(0,0,0))


def init():
    patch.set_xy(draw_triangle(0,0,0))
    ax.add_patch(patch)
    return patch,

def animate(i):
    x, y, theta = hist[i]
    patch.set_xy(draw_triangle(x, y, theta))
    return patch,

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=int(total_time/dt),
                               interval=20,
                               blit=True)
plt.ioff()
plt.show()
