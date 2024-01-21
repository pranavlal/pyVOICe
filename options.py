import getopt
import sys
import argparse

class RaspiVoiceOptions:
    def __init__(self):
        self.rows = 0
        self.columns = 0
        self.image_source = 0
        self.input_filename = ""
        self.output_filename = ""
        self.audio_card = 0
        self.volume = 0
        self.preview = False
        self.use_bw_test_image = False
        self.verbose = False
        self.negative_image = False
        self.flip = 0
        self.read_frames = 0
        self.exposure = 0
        self.brightness = 0
        self.contrast = 0.0
        self.blinders = 0
        self.zoom = 0.0
        self.foveal_mapping = False
        self.threshold = 0
        self.edge_detection_opacity = 0.0
        self.edge_detection_threshold = 0
        self.freq_lowest = 0.0
        self.freq_highest = 0.0
        self.sample_freq_Hz = 0
        self.total_time_s = 0.0
        self.use_exponential = False
        self.use_stereo = False
        self.use_delay = False
        self.use_fade = False
        self.use_diffraction = False
        self.use_bspline = False
        self.speed_of_sound_m_s = 0.0
        self.acoustical_size_of_head_m = 0.0
        self.mute = False
        self.daemon = False
        self.grab_keyboard = ""
        self.use_rotary_encoder = False
        self.speak = False
        self.quit = False

def get_default_options():
    return RaspiVoiceOptions()

cmdLineOptions = RaspiVoiceOptions()
rvopt = RaspiVoiceOptions()
rvopt_mutex = None

class MenuKeys:
    PreviousValue = 0
    PreviousOption = 1
    NextValue = 2
    NextOption = 3
    CycleValue = 4

long_getopt_options = [
    ("help", 0, "h"),
    ("daemon", 0, "d"),
    ("rows", 1, "r"),
    ("columns", 1, "c"),
    ("image_source", 1, "s"),
    ("input_filename", 1, "i"),
    ("output_filename", 1, "o"),
    ("audio_card", 1, "a"),
    ("volume", 1, "V"),
    ("preview", 0, "p"),
    ("use_bw_test_image", 1, "I"),
    ("verbose", 0, "v"),
    ("negative_image", 0, "n"),
    ("flip", 1, "f"),
    ("read_frames", 1, "R"),
    ("exposure", 1, "e"),
    ("brightness", 1, "B"),
    ("contrast", 1, "C"),
    ("blinders", 1, "b"),
    ("zoom", 1, "z"),
    ("foveal_mapping", 0, "m"),
    ("edge_detection_opacity", 1, "E"),
    ("edge_detection_threshold", 1, "G"),
    ("freq_lowest", 1, "L"),
    ("freq_highest", 1, "H"),
    ("total_time_s", 1, "t"),
    ("use_exponential", 1, "x"),
    ("use_delay", 1, "y"),
    ("use_fade", 1, "F"),
    ("use_diffraction", 1, "D"),
    ("use_bspline", 1, "N"),
    ("sample_freq_Hz", 1, "Z"),
    ("threshold", 1, "T"),
    ("use_stereo", 1, "O"),
    ("grab_keyboard", 1, "g"),
    ("use_rotary_encoder", 0, "A"),
    ("speak", 0, "S"),
]

def GetDefaultOptions():
    opt = RaspiVoiceOptions()

    opt.rows = 64
    opt.columns = 176
    opt.image_source = 1
    opt.input_filename = ""
    opt.output_filename = ""
    opt.audio_card = 0
    opt.volume = -1
    opt.preview = False
    opt.use_bw_test_image = False
    opt.verbose = False
    opt.negative_image = False
    opt.flip = 0
    opt.read_frames = 2
    opt.exposure = 0
    opt.brightness = 0
    opt.contrast = 1.0
    opt.blinders = 0
    opt.zoom = 1
    opt.foveal_mapping = False
    opt.threshold = 0
    opt.edge_detection_opacity = 0.0
    opt.edge_detection_threshold = 50
    opt.freq_lowest = 500
    opt.freq_highest = 5000
    opt.sample_freq_Hz = 48000
    opt.total_time_s = 1.05
    opt.use_exponential = True
    opt.use_stereo = True
    opt.use_delay = True
    opt.use_fade = True
    opt.use_diffraction = True
    opt.use_bspline = True
    opt.speed_of_sound_m_s = 340
    opt.acoustical_size_of_head_m = 0.20
    opt.mute = False
    opt.daemon = False
    opt.grab_keyboard = ""
    opt.use_rotary_encoder = False
    opt.speak = False

    opt.quit = False

    return opt

def SetCommandLineOptions(argv):
    global cmdLineOptions
    opt = GetDefaultOptions()

    try:
        opts, args = getopt.getopt(argv[1:], "hdr:c:s:i:o:a:V:pIvnf:R:e:B:C:b:z:mE:G:L:H:t:x:y:d:F:D:N:Z:T:O:g:AS", [option[0] for option in long_getopt_options])
    except getopt.GetoptError:
        ShowHelp()
        sys.exit(2)

    for opt, arg in opts:
        for option in long_getopt_options:
            if opt in ('-' + option[2], '--' + option[0]):
                if option[1] == 0:
                    setattr(opt, option[0], True)
                else:
                    setattr(opt, option[0], arg)
                break

    cmdLineOptions = opt
    return True

def GetCommandLineOptions():
    return cmdLineOptions

def ShowHelp():
    print("Usage: ")
    print("raspivoice {options}")
    print()
    print("Options [defaults]: ")
    print("-h, --help\t\t\t\tThis help text")
    print("-d  --daemon\t\t\t\tDaemon mode (run in the background)")
    print("-r, --rows=[64]\t\t\t\tNumber of rows, i.e. vertical (frequency) soundscape resolution (ignored if the test image is used)")
    print("-c, --columns=[178]\t\t\tNumber of columns, i.e. horizontal (time) soundscape resolution (ignored if the test image is used)")
    print("-s, --image_source=[1]\t\t\tImage source: 0 for image file, 1 for RaspiCam, 2 for 1st USB camera, 3 for 2nd USB camera...")
    print("-i, --input_filename=[]\t\t\tPath to the image file (bmp, jpg, png, ppm, tif). Reread every frame. The static test image is used if empty.")
    print("-o, --output_filename=[]\t\t\tPath to the output file (wav). Written every frame if not muted.")
    print("-a, --audio_card=[0]\t\t\tAudio card number (0,1,...), use aplay -l to get a list")
    print("-V, --volume=[-1]\t\t\tAudio volume (set by the system mixer, 0-100, -1 for no change)")
    print("-S, --speak\t\t\t\tSpeak out option changes (espeak).")
    print("-g  --grab_keyboard=[]\t\t\tGrab the keyboard device for exclusive access. Use device number(s) 0,1,2... (comma-separated without spaces) from /dev/input/event*")
    print("-A  --use_rotary_encoder\t\tUse the rotary encoder on GPIO")
    print("-p, --preview\t\t\t\tOpen preview window(s). X server required.")
    print("-v, --verbose\t\t\t\tVerbose outputs.")
    print("-n, --negative_image\t\t\tSwap bright and dark.")
    print("-f, --flip=[0]\t\t\t\t0: no flipping, 1: horizontal, 2: vertical, 3: both")
    print("-R, --read_frames=[2]\t\t\tSet the number of frames to read from the camera before processing (>= 1). Optimize for minimal lag.")
    print("-e  --exposure=[0]\t\t\tCamera exposure time setting, 1-100. Use 0 for auto.")
    print("-B  --brightness=[0]\t\t\tAdditional brightness, -255 to 255.")
    print("-C  --contrast=[1.0]\t\t\tContrast enhancement factor >= 1.0")
    print("-b  --blinders=[0]\t\t\tBlinders left and right, pixel size (0-89 for default columns)")
    print("-z  --zoom=[1.0]\t\t\tZoom factor (>= 1.0)")
    print("-m  --foveal_mapping\t\t\tEnable foveal mapping (barrel distortion magnifying center region)")
    print("-T, --threshold=[0]\t\t\tEnable the threshold for a black/white image if > 0. Range 1-255, use 127 as a starting point. 255=auto.")
    print("-E, --edge_detection_opacity=[0.0]\tEnable edge detection if > 0. Opacity of detected edges between 0.0 and 1.0.")
    print("-G  --edge_detection_threshold=[50]\tEdge detection threshold value 1-255.")
    print("-L, --freq_lowest=[500]")
    print("-H, --freq_highest=[5000]")
    print("-t, --total_time_s=[1.05]")
    print("-x  --use_exponential=[1]")
    print("-d, --use_delay=[1]")
    print("-F, --use_fade=[1]")
    print("-D  --use_diffraction=[1]")
    print("-N  --use_bspline=[1]")
    print("-Z  --sample_freq_Hz=[48000]")
    print()