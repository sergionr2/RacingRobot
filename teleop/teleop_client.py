from __future__ import print_function, with_statement, division

import zmq
import pygame
from pygame.locals import *

from constants import RASPBERRY_IP, TELEOP_PORT


UP = (1, 0)
LEFT = (0, 1)
RIGHT = (0, -1)
DOWN = (-1, 0)
STOP = (0, 0)
KEY_CODE_SPACE = 32

MAX_SPEED = 30
MAX_TURN = 45
THETA_MIN = 70
THETA_MAX = 150
STEP_SPEED = 10
STEP_TURN = 30

TELEOP_RATE = 1 / 60  # 60 fps

moveBindingsGame = {
    K_UP: UP,
    K_LEFT: LEFT,
    K_RIGHT: RIGHT,
    K_DOWN: DOWN
}


def control(x, theta, control_speed, control_turn):
    target_speed = MAX_SPEED * x
    target_turn = MAX_TURN * theta
    if target_speed > control_speed:
        control_speed = min(target_speed, control_speed + STEP_SPEED)
    elif target_speed < control_speed:
        control_speed = max(target_speed, control_speed - STEP_SPEED)
    else:
        control_speed = target_speed

    if target_turn > control_turn:
        control_turn = min(target_turn, control_turn + STEP_TURN)
    elif target_turn < control_turn:
        control_turn = max(target_turn, control_turn - STEP_TURN)
    else:
        control_turn = target_turn
    return control_speed, control_turn


def pygameMain():
    # Pygame require a window
    pygame.init()
    window = pygame.display.set_mode((800, 500), RESIZABLE)
    pygame.font.init()
    font = pygame.font.SysFont('Open Sans', 25)
    small_font = pygame.font.SysFont('Open Sans', 20)
    end = False

    def writeText(screen, text, x, y, font, color=(62, 107, 153)):
        text = str(text)
        text = font.render(text, True, color)
        screen.blit(text, (x, y))

    def clear():
        window.fill((0, 0, 0))

    def updateScreen(window, speed, turn):
        clear()
        writeText(window, 'Linear: {:.2f}, Angular: {:.2f}'.format(speed, turn), 20, 0, font, (255, 255, 255))
        help_str = 'Use arrow keys to move, q or ESCAPE to exit.'
        writeText(window, help_str, 20, 50, small_font)
        help_2 = 'space key, k : force stop ---  anything else : stop smoothly'
        writeText(window, help_2, 20, 100, small_font)

    control_speed, control_turn = 0, 0
    updateScreen(window, control_speed, control_turn)

    while not end:
        x, theta = 0, 0
        keys = pygame.key.get_pressed()
        for keycode in moveBindingsGame.keys():
            if keys[keycode]:
                x_tmp, th_tmp = moveBindingsGame[keycode]
                x += x_tmp
                theta += th_tmp

        if keys[K_k] or keys[K_SPACE]:
            x, theta = 0, 0
            control_speed, control_turn = 0, 0

        control_speed, control_turn = control(x, theta, control_speed, control_turn)
        # Send Orders
        angle_order = sendToServer(socket, control_speed, control_turn)

        updateScreen(window, control_speed, angle_order)

        for event in pygame.event.get():
            if event.type == QUIT or event.type == KEYDOWN and event.key in [K_ESCAPE, K_q]:
                end = True
        pygame.display.flip()
        # Limit FPS
        pygame.time.Clock().tick(1 / TELEOP_RATE)


def sendToServer(socket, control_speed, control_turn):
    """
    :param socket: (zmq socket object)
    :param control_speed: (float)
    :param control_turn: (float)
    """
    # Send Orders
    t = (control_turn + MAX_TURN) / (2 * MAX_TURN)
    angle_order = int(THETA_MIN * t + THETA_MAX * (1 - t))
    socket.send_json((control_speed, angle_order))
    return angle_order


if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    print("Connecting to ... {}".format(RASPBERRY_IP))
    socket.connect("tcp://{}:{}".format(RASPBERRY_IP, TELEOP_PORT))

    msg = socket.recv()
    print("Connected To Server")
    try:
        pygameMain()
    except KeyboardInterrupt as e:
        pass
    finally:
        socket.send_json((-999, -999))
        socket.close()
