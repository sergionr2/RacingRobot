from __future__ import print_function, with_statement, division

import zmq

from teleop import *


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

    x, theta, status, count = 0, 0, 0, 0
    control_speed, control_turn = 0, 0
    angle_order = 0
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


host = "192.168.12.252"
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://{}:{}".format(host, port))

msg = socket.recv()
print("Connected To Server")
try:
    pygameMain()
except KeyboardInterrupt as e:
    pass
finally:
    socket.send_json((-999, -999))
    socket.close()
