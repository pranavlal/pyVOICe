import os
import fcntl
import struct
import array
import sys
import termios
import tty
import curses
import time
from enum import Enum
from evdev import InputDevice, categorize, ecodes
import math

class InputType(Enum):
    Terminal = 0
    NCurses = 1
    Keyboard = 2
    RotaryEncoder = 3

class MenuKeys(Enum):
    PreviousOption = ord('w')
    NextOption = ord('s')
    PreviousValue = ord('a')
    NextValue = ord('d')
    CycleValue = ord('c')

class Encoder:
    def __init__(self, pin_clk, pin_dt, pin_sw):
        self.pin_clk = pin_clk
        self.pin_dt = pin_dt
        self.pin_sw = pin_sw
        self.value = 0
        self.switchpresscount = 0

        GPIO.setup(self.pin_clk, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.pin_dt, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.pin_sw, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        GPIO.add_event_detect(self.pin_clk, GPIO.BOTH, callback=self.update, bouncetime=2)
        GPIO.add_event_detect(self.pin_sw, GPIO.FALLING, callback=self.switch_pressed, bouncetime=300)

    def update(self, channel):
        clk_state = GPIO.input(self.pin_clk)
        dt_state = GPIO.input(self.pin_dt)

        if clk_state == dt_state:
            self.value += 1
        else:
            self.value -= 1

    def switch_pressed(self, channel):
        self.switchpresscount += 1

class KeyboardInput:
    def __init__(self):
        self.input_type = InputType.Terminal
        self.fevdev = []
        self.current_option_index = 0
        self.last_encoder_value = 0
        self.last_switch_press_count = 0
        self.encoder = None
        self.verbose = False

    def setup_rotary_encoder(self):
        GPIO.setmode(GPIO.BCM)
        self.encoder = Encoder(4, 5, 6)

    def read_rotary_encoder(self):
        ch = -1
        if self.encoder:
            time.sleep(0.1)
            value = self.encoder.value
            if value != self.last_encoder_value:
                if self.verbose:
                    print("\nlast_encoder_value: {}, new encoder value: {}\n".format(self.last_encoder_value, value))

                if value > self.last_encoder_value:
                    ch = MenuKeys.NextOption.value
                elif value < self.last_encoder_value:
                    ch = MenuKeys.PreviousOption.value

                self.last_encoder_value = value

            switch_press_count = self.encoder.switchpresscount
            if switch_press_count != self.last_switch_press_count:
                if self.verbose:
                    print("\nlast_switch_press_count: {}, new switchpresscount: {}\n".format(self.last_switch_press_count, switch_press_count))
                ch = MenuKeys.CycleValue.value
                self.last_switch_press_count = switch_press_count

        return ch

    def set_input_type(self, input_type, keyboard=None):
        self.input_type = input_type

        if input_type == InputType.Keyboard:
            if not self.grab_keyboard(keyboard):
                return False
        elif input_type == InputType.RotaryEncoder:
            self.setup_rotary_encoder()

        return True

    def grab_keyboard(self, keyboard):
        try:
            for event_device_id in keyboard.split(','):
                device_id = int(event_device_id)
                devpath = "/dev/input/event" + event_device_id

                fevdev = open(devpath, 'rb')
                self.fevdev.append(fevdev)

                if fcntl.ioctl(fevdev, termios.EVIOCGRAB, 1) != 0:  # grab keyboard
                    return False

            return True
        except IOError as e:
            print("Error opening/grabbing keyboard: {}".format(e))
            return False

    def close_keyboard(self):
        for fevdev in self.fevdev:
            try:
                fcntl.ioctl(fevdev, termios.EVIOCGRAB, 0)  # Release grabbed keyboard
                fevdev.close()
            except IOError as e:
                print("Error closing keyboard: {}".format(e))

    def read_key(self):
        if self.input_type == InputType.Terminal:
            return sys.stdin.read(1)
        elif self.input_type == InputType.NCurses:
            return stdscr.getch()
        elif self.input_type == InputType.Keyboard:
            for fevdev in self.fevdev:
                if fevdev == -1:
                    return -1
                ev = array.array('I', [0, 0, 0, 0, 0, 0, 0])
                rd = fevdev.readinto(ev)
                if rd < 0:
                    continue

                value = ev[6]
                if value != ord(' ') and ev[4] == 1 and ev[1] == 1:  # value=1: key press, value=0 key release
                    return value
            return -1
        elif self.input_type == InputType.RotaryEncoder:
            return self.read_rotary_encoder()

    def get_interactive_command_list(self):
        cmdlist = [
            "raspivoice",
            "Press key to cycle settings:",
            "0: Mute [off, on]",
            "1: Negative image [off, on]",
            "2: Zoom [off, x2, x4]",
            "3: Blinders [off, on]",
            "4: Edge detection [off, 50%%, 100%%]",
            "5: Threshold [off, 25%%, 50%%, 75%%]",
            "6: Brightness [low, normal, high]",
            "7: Contrast [x1, x2, x3]",
            "8: Foveal mapping [off, on]",
            ".: Restore defaults",
            "q, [Escape]: Quit"
        ]

        return '\n'.join(cmdlist)
    def key_pressed_action(self, ch):
        option_cycle = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '.']
        option_changed = False
        changevalue = 0  # -1: decrease value, 0: no change, 1: increase value, 2: cycle values
        state_str = ""
        newvolume = -1

        if self.verbose:
            print("KeyPressedAction: ch =", ch)

        # Menu navigation keys:
        if ch == MenuKeys.PreviousValue:
            changevalue = -1
        elif ch == MenuKeys.PreviousOption:
            option_changed = True
            if self.current_option_index > 0:
                self.current_option_index -= 1
            else:
                self.current_option_index = len(option_cycle) - 1
        elif ch == MenuKeys.NextValue:
            changevalue = 1
        elif ch == MenuKeys.NextOption:
            option_changed = True
            if self.current_option_index < (len(option_cycle) - 1):
                self.current_option_index += 1
            else:
                self.current_option_index = 0
        elif ch == MenuKeys.CycleValue:
            changevalue = 2

        if option_changed:
            # just speak out currently selected option
            current_option = option_cycle[self.current_option_index]
            state_str = self.get_option_state_str(current_option)

        else:  # value change requested
            if changevalue != 0:
                # use current option if navigation keys were used:
                ch = option_cycle[self.current_option_index]
            else:
                changevalue = 2  # cycle value if direct key was used

            # change value and speak out new value:
            with self.rvopt_mutex:
                state_str = self.change_option_value(ch, changevalue)

        return state_str

    def key_event_map(self, event_code):
        if self.verbose:
            # event codes see linux/input.h header file
            print("KeyEventMap: event_code =", event_code)

        ch = 0
        if event_code in [ecodes.KEY_A, ecodes.KEY_LEFT, ecodes.KEY_BACK, ecodes.KEY_STOP, ecodes.KEY_F1]:
            ch = MenuKeys.PreviousValue
        elif event_code in [ecodes.KEY_S, ecodes.KEY_DOWN, ecodes.KEY_FORWARD, ecodes.KEY_NEXTSONG, ecodes.KEY_F3]:
            ch = MenuKeys.NextOption
        elif event_code in [ecodes.KEY_D, ecodes.KEY_RIGHT, ecodes.KEY_PLAYPAUSE, ecodes.KEY_PLAY, ecodes.KEY_PAUSE, ecodes.BTN_RIGHT]:
            ch = MenuKeys.NextValue
        elif event_code in [ecodes.KEY_W, ecodes.KEY_UP, ecodes.KEY_PREVIOUSSONG, ecodes.KEY_REWIND, ecodes.KEY_HOME]:
            ch = MenuKeys.PreviousOption
        elif event_code in [ecodes.KEY_LINEFEED, ecodes.KEY_KPENTER, ecodes.KEY_SPACE, ecodes.BTN_LEFT]:
            ch = MenuKeys.CycleValue
        elif event_code in [ecodes.KEY_0, ecodes.KEY_KP0, ecodes.KEY_MUTE, ecodes.KEY_NUMERIC_0]:
            ch = ord('0')
        elif event_code in [ecodes.KEY_1, ecodes.KEY_KP1]:
            ch = ord('1')
        elif event_code in [ecodes.KEY_2, ecodes.KEY_KP2]:
            ch = ord('2')
        elif event_code in [ecodes.KEY_3, ecodes.KEY_KP3]:
            ch = ord('3')
        elif event_code in [ecodes.KEY_4, ecodes.KEY_KP4]:
            ch = ord('4')
        elif event_code in [ecodes.KEY_5, ecodes.KEY_KP5]:
            ch = ord('5')
        elif event_code in [ecodes.KEY_6, ecodes.KEY_KP6]:
            ch = ord('6')
        elif event_code in [ecodes.KEY_7, ecodes.KEY_KP7]:
            ch = ord('7')
        elif event_code in [ecodes.KEY_8, ecodes.KEY_KP8]:
            ch = ord('8')
        elif event_code in [ecodes.KEY_9, ecodes.KEY_KP9]:
            ch = ord('9')
        elif event_code in [ecodes.KEY_NUMERIC_STAR, ecodes.KEY_KPASTERISK]:
            ch = ord('*')
        elif event_code in [ecodes.KEY_SLASH, ecodes.KEY_KPSLASH]:
            ch = ord('/')
        elif event_code in [ecodes.KEY_EQUAL, ecodes.KEY_KPEQUAL]:
            ch = ord('=')
        elif event_code in [ecodes.KEY_DOT, ecodes.KEY_KPDOT, ecodes.KEY_KPCOMMA, ecodes.KEY_KPJPCOMMA, ecodes.KEY_BACKSPACE, ecodes.KEY_DC]:
            ch = ord('.')
        elif event_code in [ecodes.KEY_KPPLUS, ecodes.KEY_VOLUMEUP]:
            ch = ord('+')
        elif event_code in [ecodes.KEY_MINUS, ecodes.KEY_KPMINUS, ecodes.KEY_VOLUMEDOWN]:
            ch = ord('-')

        return ch

    def change_index(self, i, maxindex, changevalue):
        new_index = 0

        if changevalue == -1:  # decrease (stop at 0)
            new_index = i - 1
            if new_index < 0:
                new_index = 0
        elif changevalue == 1:  # increase (stop at max)
            new_index = i + 1
            if new_index > maxindex:
                new_index = maxindex
        elif changevalue == 2:  # increase (continue at 0)
            new_index = i + 1
            if new_index > maxindex:
                new_index = 0
        elif changevalue == -2:  # decrease (continue at max)
            new_index = i - 1
            if new_index < 0:
                new_index = maxindex

        return new_index

    def cycle_values(self, current_value, value_list, changevalue):
        for i, value in enumerate(value_list):
            if math.isclose(current_value, value):
                current_value = value_list[self.change_index(i, len(value_list) - 1, changevalue)]
                return
        current_value = value_list[0]

    def get_option_state_str(self, option):
        state_str = ""
        if option == '0':
            state_str = "mute"
        elif option == '1':
            state_str = "negative image"
        elif option == '2':
            state_str = "zoom factor " + str(self.rvopt.zoom)
        elif option == '3':
            state_str = "blinders " + ("off" if self.rvopt.blinders == 0 else "on")
        elif option == '4':
            state_str = "edge detection " + str(self.rvopt.edge_detection_opacity)
        elif option == '5':
            state_str = "threshold " + str(self.rvopt.threshold)
        elif option == '6':
            state_str = "brightness " + ("high" if self.rvopt.brightness == 100 else "medium" if self.rvopt.brightness == 0 else "low")
        elif option == '7':
            state_str = "contrast factor " + str(self.rvopt.contrast)
        elif option == '8':
            state_str = "foveal mapping " + ("on" if self.rvopt.foveal_mapping else "off")
        elif option == '.':
            state_str = "default options"
        elif option == 'q':
            state_str = "quit"

        return state_str
