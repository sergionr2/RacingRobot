import sys, select, termios, tty
import time
import curses

msg = """
---------------------------
Moving around:
        i
   j    k    l
        ,

space key, k : force stop
anything else : stop smoothly

CTRL-C to quit
"""

class TextWindow:
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

def publish(interface):
    interface.clear()
    interface.writeLine(2, 'Linear: {}, Angular: {}'.format(0, 0))
    interface.writeLine(3, 'Use arrow keys to move, space to stop, q or CTRL-C to exit.')
    interface.refresh()

# stdscr: main window
def test(stdscr):
    interface = TextWindow(stdscr)
    while True:
        keycode = interface.readKey()
        if keycode is not None:
            publish(interface)
            if keycode == curses.KEY_DOWN:
                curses.beep()
            elif keycode == ord('q'):
                break

class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

UP = (1,0)
LEFT = (0,1)
RIGHT = (0,-1)
DOWN = (-1,0)
STOP = (0,0)

moveBindings = {
                'i':UP,
                'j': LEFT,
                'l': RIGHT,
                ',':DOWN
                }

moveBindings = {
                curses.KEY_UP:UP,
                curses.KEY_LEFT: LEFT,
                curses.KEY_RIGHT: RIGHT,
                curses.KEY_DOWN:DOWN
                }
def getKey():
    # file descriptor
    fd = sys.stdin.fileno()
    tty.setraw(fd)
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    # rlist: wait until ready for reading
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

MAX_SPEED = .2
MAX_TURN = 1

if __name__=="__main__":
    try:
        curses.wrapper(test)
    except KeyboardInterrupt:
        exit()
    settings = termios.tcgetattr(sys.stdin)
    x = 0
    theta = 0
    status = 0
    count = 0
    target_speed, target_turn = 0, 0
    control_speed, control_turn = 0, 0
    start_time = time.time()
    counter = 0
    try:
        print(msg)
        while True:
            key = getKey()
            counter += 1
            if counter % 100 == 0:
                print("{:.2f} fps".format(counter / (time.time() - start_time)))
            if key in moveBindings.keys():
                # print(key)
                x = moveBindings[key][0]
                theta = moveBindings[key][1]
                count = 0
            elif key == ' ' or key == 'k' :
                x = 0
                theta = 0
                control_speed = 0
                control_turn = 0
                # print('STOP')
            else:
                count += 1
                if count > 4:
                    x = 0
                    theta = 0
                if (key == '\x03'):
                    break

            target_speed = MAX_SPEED * x
            target_turn = MAX_TURN * theta

            # Smooth control
            if target_speed > control_speed:
                control_speed = min(target_speed, control_speed + 0.02)
            elif target_speed < control_speed:
                control_speed = max(target_speed, control_speed - 0.02)
            else:
                control_speed = target_speed

            if target_turn > control_turn:
                control_turn = min(target_turn, control_turn + 0.1)
            elif target_turn < control_turn:
                control_turn = max(target_turn, control_turn - 0.1)
            else:
                control_turn = target_turn
    except Exception as e:
        raise e

    finally:
        pass

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
