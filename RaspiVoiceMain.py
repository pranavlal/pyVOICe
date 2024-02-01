import os
import sys
import threading
import time
import curses

from options import cmdLineOptions, SetCommandLineOptions, GetCommandLineOptions, RaspiVoiceOptions
from RaspiVoice import RaspiVoice
from KeyboardInput import KeyboardInput
from AudioData import AudioData

exc_ptr = None
rvopt_mutex = threading.Lock()
cmdline_opt = RaspiVoiceOptions()
pidfilename = "/var/run/raspivoice/raspivoice.pid"

newfd = -1
fd = None
scr = None
saved_stdout = -1
saved_stderr = -1


def run_worker_thread():
    global exc_ptr
    try:
        # Init:
        raspi_voice = RaspiVoice(cmdline_opt)

        while not rvopt_local.quit:
            # Read one frame:
            raspi_voice.GrabAndProcessFrame(rvopt_local)

            # Copy any new options:
            with rvopt_mutex:
                rvopt_local = cmdLineOptions

            # Play frame:
            raspi_voice.PlayFrame(rvopt_local)
    except RuntimeError as err:
        exc_ptr = sys.exc_info()[1]
        if rvopt_local.verbose:
            print(err)
        with rvopt_mutex:
            cmdLineOptions.quit = True


def setup_screen():
    # ncurses screen setup:
    # initscr()
    global newfd, fd, scr, saved_stdout, saved_stderr

    fd = open("/dev/tty", "r+")
    if fd is None:
        print("Cannot open screen.")
        return False

    scr = curses.newterm(None, fd, fd)
    if scr is None:
        print("Cannot open screen.")
        close_screen()
        return False

    newfd = os.open("/dev/null", os.O_WRONLY)
    if newfd == -1:
        print("Cannot open screen.")
        close_screen()
        return False

    sys.stdout.flush()
    sys.stderr.flush()

    saved_stdout = os.dup(sys.stdout.fileno())
    saved_stderr = os.dup(sys.stderr.fileno())

    os.dup2(newfd, sys.stdout.fileno())
    os.dup2(newfd, sys.stderr.fileno())
    sys.stdout.reconfigure(line_buffering=True)

    curses.clear()
    curses.noecho()
    curses.cbreak()
    curses.keypad(curses.stdscr, True)
    curses.timeout(10)  # ms

    return True


def close_screen():
    # quit ncurses:
    curses.refresh()
    curses.endwin()

    if saved_stdout != -1:
        os.dup2(saved_stdout, sys.stdout.fileno())
        os.close(saved_stdout)
    if saved_stderr != -1:
        os.dup2(saved_stderr, sys.stderr.fileno())
        os.close(saved_stderr)

    if newfd != -1:
        os.close(newfd)
    if scr is not None:
        curses.delscreen(scr)
    if fd is not None:
        fd.close()


def main_loop(keyboard_input):
    quit_flag = False
    audio_data = AudioData(cmdline_opt.audio_card)

    while not quit_flag:
        ch = keyboard_input.ReadKey()

        if ch != curses.ERR:
            print("ch:", ch)

            state_str = keyboard_input.KeyPressedAction(ch)

            with rvopt_mutex:
                if quit_flag or cmdline_opt.quit:
                    quit_flag = True
                    cmdline_opt.quit = True

                # Volume change?
                if cmdline_opt.volume != -1:
                    audio_data.SetVolume(cmdline_opt.volume)

            # Speak state_str?
            if cmdline_opt.speak and state_str:
                if not audio_data.Speak(state_str):
                    print("Error calling Speak(). Use verbose mode for more info.")


def daemon_startup():
    pid = os.fork()
    if pid < 0:
        sys.exit(EXIT_FAILURE)
    if pid > 0:
        sys.exit(EXIT_SUCCESS)

    os.umask(0)

    sid = os.setsid()
    if sid < 0:
        sys.exit(EXIT_FAILURE)

    if os.chdir("/") < 0:
        sys.exit(EXIT_FAILURE)

    os.close(sys.stdout.fileno())
    os.close(sys.stderr.fileno())

    # write pidfile if possible:
    try:
        with open(pidfilename, "wt") as fp_pid:
            fp_pid.write(f"{os.getpid()}\n")
    except IOError:
        sys.exit(EXIT_FAILURE)


if __name__ == "__main__":
    if not SetCommandLineOptions(sys.argv):
        sys.exit(-1)

    cmdline_opt = GetCommandLineOptions()

    if cmdline_opt.daemon:
        print("raspivoice daemon started.")
        daemon_startup()

    rvopt_mutex = threading.Lock()
    rvopt_local = cmdline_opt

    # Setup keyboard:
    keyboard_input = KeyboardInput()
    keyboard_input.Verbose = cmdline_opt.verbose

    use_ncurses = not (cmdline_opt.verbose or cmdline_opt.daemon)

    if cmdline_opt.use_rotary_encoder:
        keyboard_input.SetInputType(KeyboardInput.InputType.RotaryEncoder)
    elif cmdline_opt.grab_keyboard != "":
        if not keyboard_input.SetInputType(KeyboardInput.InputType.Keyboard, cmdline_opt.grab_keyboard):
            print(f"Cannot grab keyboard device: {cmdline_opt.grab_keyboard}.")
            sys.exit(-1)
    elif use_ncurses:
        keyboard_input.SetInputType(KeyboardInput.InputType.NCurses)
    elif not cmdline_opt.daemon:
        keyboard_input.SetInputType(KeyboardInput.InputType.Terminal)

    # Start Program in worker thread:
    # Warning: Do not read or write rvopt or quit_flag without locking after this.
    thr = threading.Thread(target=run_worker_thread)
    AudioData.Init()
    thr.start()

    # Setup UI:
    if use_ncurses:
        # Show interactive screen:
        if setup_screen():
            curses.printw(f"{keyboard_input.GetInteractiveCommandList()}")
            curses.refresh()

            main_loop(keyboard_input)

            close_screen()
    elif cmdline_opt.verbose and not cmdline_opt.daemon:
        print("Verbose mode on, curses UI disabled.")
       

print(keyboard_input.GetInteractiveCommandList())
if True:
        main_loop(keyboard_input)
else:
        main_loop(keyboard_input)

if cmdline_opt.grab_keyboard != "":
        keyboard_input.ReleaseKeyboard()

    # Wait for worker thread:
thr.join()

    # Check for exception from worker thread:
if exc_ptr is not None:
        exc_type, exc_value, _ = exc_ptr
        print(f"Error: {exc_value}")
        sys.exit(-1)
