import curses
import time

UP = (1,0)
LEFT = (0,1)
RIGHT = (0,-1)
DOWN = (-1,0)
STOP = (0,0)
KEY_CODE_SPACE = 32

MAX_SPEED = 0.2
MAX_TURN = 1
STEP_SPEED = 0.02
STEP_TURN = 0.01

moveBindings = {
                curses.KEY_UP: UP,
                curses.KEY_LEFT: LEFT,
                curses.KEY_RIGHT: RIGHT,
                curses.KEY_DOWN: DOWN
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

# stdscr: main window
def main(stdscr):
    interface = Interface(stdscr)
    x, theta, status, count = 0, 0, 0, 0
    target_speed, target_turn = 0, 0
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

        target_speed = MAX_SPEED * x
        target_turn = MAX_TURN * theta

        # Smooth control
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
        # force 30 fps
        time.sleep(1/30)

if __name__=="__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        exit()
