import cv2
import numpy as np

from ImageToSoundscape import ImageToSoundscapeConverter
from AudioData import AudioData

class RaspiVoice:
    def __init__(self, opt):
        self.rows = opt.rows
        self.columns = opt.columns
        self.image_source = opt.image_source
        self.preview = opt.preview
        self.use_bw_test_image = opt.use_bw_test_image
        self.verbose = opt.verbose
        self.opt = opt

        if self.image_source == 0 and not opt.input_filename:
            self.rows = 64
            self.columns = 64

        self.init()

        self.ImgTossConverter = ImageToSoundscapeConverter(
            self.rows, self.columns, opt.freq_lowest, opt.freq_highest,
            opt.sample_freq_Hz, opt.total_time_s, opt.use_exponential,
            opt.use_stereo, opt.use_delay, opt.use_fade, opt.use_diffraction,
            opt.use_bspline, opt.speed_of_sound_m_s, opt.acoustical_size_of_head_m
        )

    def __del__(self):
        if self.i2ssConverter:
            del self.i2ssConverter

        if self.image_source == 1:
            self.raspiCam.release()

    def init(self):
        if self.verbose:
            print('Printing time')

        self.image = np.zeros((self.rows, self.columns), dtype=np.float32)

        if self.image_source == 0:
            if not self.opt.input_filename:
                self.init_test_image()
            else:
                self.init_file_image()

        elif self.image_source == 1:
            self.init_raspi_cam()
        elif self.image_source >= 2:
            self.init_usb_cam()

        if self.preview:
            cv2.namedWindow("RaspiVoice Preview", cv2.WINDOW_NORMAL)

        im = self.read_image()
        self.process_image(im)

    def init_test_image(self):
        if self.verbose:
            print("Using B/W test image" if self.use_bw_test_image else "Using grayscale test image")

        for j in range(self.columns):
            for i in range(self.rows):
                if self.use_bw_test_image:
                    self.image[i, j] = 1.0 if P_bw[self.rows - i - 1][j] != '#' else 0.0
                else:
                    val = pow(10.0, (P_grayscale[i][j] - ord('a') - 15) / 10.0) if P_grayscale[i][j] > 'a' else 0.0
                    self.image[i, j] = val

    def init_file_image(self):
        if self.verbose:
            print(f"Checking input file {self.opt.input_filename}")

        mat = cv2.imread(self.opt.input_filename, cv2.IMREAD_GRAYSCALE)
        if mat is None:
            raise RuntimeError("Cannot read input file as image.")
        print("ok")

    def init_raspi_cam(self):
        if self.verbose:
            print("Opening RaspiCam...")

        self.raspiCam = cv2.VideoCapture(0)
        self.raspiCam.set(cv2.CAP_PROP_FORMAT, cv2.CV_8UC1)
        self.raspiCam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.raspiCam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        if 1 <= self.opt.exposure <= 100:
            self.raspiCam.set(cv2.CAP_PROP_EXPOSURE, self.opt.exposure)

        if not self.raspiCam.isOpened():
            raise RuntimeError("Error opening RaspiCam.")
        elif self.verbose:
            print("Ok")

    def init_usb_cam(self):
        cam_id = self.image_source - 2

        if self.verbose:
            print("Opening USB camera...")

        self.cap = cv2.VideoCapture(cam_id)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        if 1 <= self.opt.exposure <= 100:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.opt.exposure)

        if self.verbose:
            print("Ok")

    def read_image(self):
        raw_image = None
        processed_image = None

        if self.verbose:
            print("ReadImage start")

        if self.image_source == 0 and self.opt.input_filename:
            raw_image = cv2.imread(self.opt.input_filename, cv2.IMREAD_GRAYSCALE)
            processed_image = raw_image.copy()
        elif self.image_source == 1:
            self.raspiCam.grab()
            _, raw_image = self.raspiCam.retrieve()
            if self.opt.read_frames > 1:
                for _ in range(1, self.opt.read_frames):
                    self.raspiCam.grab()
                    _, raw_image = self.raspiCam.retrieve()
            processed_image = raw_image.copy()
        elif self.image_source >= 2:
            _, raw_image = self.cap.read()
            if self.opt.read_frames > 1:
                for _ in range(1, self.opt.read_frames):
                    _, raw_image = self.cap.read()
            processed_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

        return processed_image

    def process_image(self, raw_image):
        processed_image = raw_image.copy()

        if self.verbose:
            print("ProcessImage start")

        if self.image_source > 0 or self.opt.input_filename:
            if self.opt.foveal_mapping:
                camera_matrix = np.array([[100, 0, processed_image.shape[1] // 2],
                                          [0, 100, processed_image.shape[0] // 2],
                                          [0, 0, 1]], dtype=np.float32)
                dist_coeffs = np.array([5.0, 5.0, 0, 0], dtype=np.float32)
                processed_image = cv2.undistort(processed_image, camera_matrix, dist_coeffs)

            if self.opt.zoom > 1.0:
                h, w = processed_image.shape
                z = self.opt.zoom
                roi = ((w / 2.0) - w / (2.0 * z), (h / 2.0) - h / (2.0 * z), w / z, h / z)
                processed_image = processed_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

            if processed_image.shape[0] != self.rows or processed_image.shape[1] != self.columns:
                processed_image = cv2.resize(processed_image, (self.columns, self.rows))

            if 0 < self.opt.blinders < self.columns / 2:
                processed_image[:, :int(self.opt.blinders)] = 0
                processed_image[:, -int(self.opt.blinders):] = 0

            if self.opt.contrast != 1.0 or self.opt.brightness != 0:
                alpha = self.opt.contrast
                beta = self.opt.brightness
                processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)

            if self.opt.threshold > 0:
                if self.opt.threshold < 255:
                    _, processed_image = cv2.threshold(processed_image, self.opt.threshold, 255, cv2.THRESH_BINARY)
                else:
                    _, processed_image = cv2.threshold(processed_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            if self.opt.negative_image:
                processed_image = 255 - processed_image

            if 0 < self.opt.edge_detection_opacity <= 1.0:
                blur_image = cv2.blur(processed_image, (3, 3))
                edges_image = cv2.Canny(blur_image, self.opt.edge_detection_threshold,
                                        self.opt.edge_detection_threshold * 3, 3)
                processed_image = cv2.addWeighted(edges_image, self.opt.edge_detection_opacity,
                                                  processed_image, 1.0 - self.opt.edge_detection_opacity, 0.0)

            if 1 <= self.opt.flip <= 3:
                flip_code = 1 if self.opt.flip == 1 else 0 if self.opt.flip == 2 else -1
                processed_image = cv2.flip(processed_image, flip_code)

            if self.preview:
                cv2.imshow("RaspiVoice Preview", processed_image)
                cv2.waitKey(200)

            for j in range(self.columns):
                for i in range(self.rows):
                    m_val = processed_image[self.rows - 1 - i, j] // 16
                    self.image[i, j] = 0 if m_val == 0 else pow(10.0, (m_val - 15) / 10.0)

    def grab_and_process_frame(self, opt):
        self.opt = opt
        im = self.read_image()
        self.process_image(im)

        if self.verbose:
            print("vOICe algorithm process start")

        self.i2ssConverter.process(self.image)

    def play_frame(self, opt):
        if opt.quit:
            return

        if not opt.mute:
            audio_data = self.i2ssConverter.get_audio_data()
            audio_data.card_number = opt.audio_card
            audio_data.verbose = self.verbose

            if self.verbose:
                print("Playing audio")

            audio_data.play()

            if opt.output_filename:
                audio_data.save_to_wav_file(opt.output_filename)

        elif self.verbose:
            print("Muted, not playing audio")