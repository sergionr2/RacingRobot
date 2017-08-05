from __future__ import print_function, with_statement, division

import curses
import time

import pygame
from pygame.locals import *

import common
from common import *
from listener import decodeOrder

UP = (1,0)
LEFT = (0,1)
RIGHT = (0,-1)
DOWN = (-1,0)
STOP = (0,0)
KEY_CODE_SPACE = 32

MAX_SPEED = 100
MAX_TURN = 45
THETA_MIN = 60
THETA_MAX = 130
STEP_SPEED = 10
STEP_TURN = 30

moveBindings = {
                curses.KEY_UP: UP,
                curses.KEY_LEFT: LEFT,
                curses.KEY_RIGHT: RIGHT,
                curses.KEY_DOWN: DOWN
                }

moveBindingsGame = {
                K_UP: UP,
                K_LEFT: LEFT,
                K_RIGHT: RIGHT,
                K_DOWN: DOWN
                }

class Interface:
    def __init__(self, stdscr, lines=10):
        self._screen = stdscr
        self._screen.nodelay(True)
        # Reset cursor
        curses.curs_set(0)
        self._num_lines = lines

    def readKey(self):
        keycode = self._screen.getch()
        return keycode if keycode != -1 else None

    def clear(self):
        self._screen.clear()

    def writeLine(self, line_num, message):
        if line_num < 0 or line_num >= self._num_lines:
            raise ValueError('line_num out of bounds')
        height, width = self._screen.getmaxyx()
        y = (height // self._num_lines) * line_num
        x = 10
        for text in message.split('\n'):
            text = text.ljust(width)
            self._screen.addstr(y, x, text)
            y += 1

    def refresh(self):
        self._screen.refresh()

def publish(interface, speed, turn, info):
    interface.clear()
    interface.writeLine(2, 'Linear: {:.2f}, Angular: {:.2f}'.format(speed, turn))
    interface.writeLine(3, 'Use arrow keys to move, space to stop, q or CTRL-C to exit.')
    interface.writeLine(4, 'space key, k : force stop ---  anything else : stop smoothly')
    interface.writeLine(5, info)
    interface.refresh()

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

# stdscr: main window
def main(stdscr):
    interface = Interface(stdscr)
    x, theta, status, count = 0, 0, 0, 0
    control_speed, control_turn = 0, 0
    start_time = time.time()
    counter = 0
    while True:
        keycode = interface.readKey()
        counter += 1
        info = "{:.2f} fps".format(counter/(time.time()-start_time))

        publish(interface, control_speed, control_turn, info)
        if keycode in moveBindings.keys():
            x, theta = moveBindings[keycode]
            count = 0
        elif keycode == ord('k') or keycode == KEY_CODE_SPACE:
                x, theta = 0, 0
                control_speed, control_turn = 0, 0
        elif keycode == ord('q'):
            break
        else:
            count += 1
            if count > 4:
                x, theta = 0, 0

        # Smooth control
        control_speed, control_turn = control(x, theta, control_speed, control_turn)
        # force 30 fps
        time.sleep(1/30)

def pygameMain():
    # Pygame require a window
    pygame.init()
    window = pygame.display.set_mode((800,500), RESIZABLE)
    pygame.font.init()
    font = pygame.font.SysFont('Open Sans', 25)
    small_font = pygame.font.SysFont('Open Sans', 20)
    end = False

    def writeText(screen, text, x, y, font, color=(62, 107, 153)):
        text = str(text)
        text = font.render(text, True, color)
        screen.blit(text, (x, y))

    def clear():
        window.fill((0,0,0))

    def updateScreen(window, speed, turn):
        clear()
        writeText(window, 'Linear: {:.2f}, Angular: {:.2f}'.format(speed, turn), 20, 0, font, (255, 255, 255))
        help_str =  'Use arrow keys to move, q or ESCAPE to exit.'
        writeText(window, help_str, 20, 50, small_font)
        help_2 =  'space key, k : force stop ---  anything else : stop smoothly'
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
        common.command_queue.put((Order.MOTOR, control_speed))
        t = (control_turn + MAX_TURN) / (2 * MAX_TURN)
        angle_order = int(THETA_MIN  * t + THETA_MAX * (1 - t))
        common.command_queue.put((Order.SERVO, angle_order))

        updateScreen(window, control_speed, angle_order)

        for event in pygame.event.get():
            if event.type == QUIT or event.type == KEYDOWN and event.key in [K_ESCAPE, K_q]:
                end = True
        pygame.display.flip()
        # force 30 fps
        pygame.time.Clock().tick(1/common.rate)

if __name__=="__main__":
    # Does not handle multiple key pressed
    # try:
    #     curses.wrapper(main)
    # except KeyboardInterrupt:
    #     exit()
    # pygameMain()
    try:
        serial_port = get_serial_ports()[0]
        serial_file = serial.Serial(port=serial_port, baudrate=BAUDRATE, timeout=0, writeTimeout=0)
    except Exception as e:
        raise e
    # serial_file_write = open("test.log", 'ab')
    # serial_file_read = open("test.log", 'rb')

    while not is_connected:
        print("Waiting for arduino...")
        sendOrder(serial_file, Order.HELLO.value)
        bytes_array = bytearray(serial_file.read(1))
        if not bytes_array:
            time.sleep(2)
            continue
        byte = bytes_array[0]
        if byte in [Order.HELLO.value, Order.ALREADY_CONNECTED.value]:
            is_connected = True
        time.sleep(1)

    threads = [CommandThread(serial_file, command_queue),
               ListenerThread(serial_file)]
    for t in threads:
        t.start()

    pygameMain()
    common.exit_signal = True
    n_received_semaphore.release()
    print("EXIT")

    for t in threads:
        t.join()
