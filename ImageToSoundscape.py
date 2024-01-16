import numpy as np
from scipy.io.wavfile import write
from scipy.signal import bspline

class ImageToSoundscapeConverter:
    def __init__(self, rows, columns, freq_lowest, freq_highest, sample_freq_Hz,
                 total_time_sec, use_exponential, use_stereo, use_delay, use_fade,
                 use_diffraction, use_bspline, speed_of_sound_ms, acoustical_size_of_head_m):
        self.rows = rows
        self.columns = columns
        self.freq_highest = freq_highest
        self.freq_lowest = freq_lowest
        self.sample_freq_Hz = sample_freq_Hz
        self.total_time_sec = total_time_sec
        self.use_exponential = use_exponential
        self.use_stereo = use_stereo
        self.use_delay = use_delay
        self.use_fade = use_fade
        self.use_diffraction = use_diffraction
        self.use_bspline = use_bspline
        self.speed_of_sound_ms = speed_of_sound_ms
        self.acoustical_size_of_head_m  = acoustical_size_of_head_m
        self.sample_count = 2 * int(0.5 * sample_freq_Hz * total_time_sec)
        self.sample_per_column = int(self.sample_count/columns)
        self.scale = 0.5 / np.sqrt(rows)
        
        self.audio_data = np.zeros((self.sample_count, 2), dtype=np.int16)
        self.omega = np.zeros(rows)
        self.phi0 = np.zeros(rows)
        self.wavefrom_cache_left_channel = np.zeros((self.sample_count, rows))
        self.wavefrom_cache_right_channel = np.zeros((self.sample_count, rows))
        
        self._initialize()
        
    def _initialize(self):
         #Set lin|exp (0|1) frequency distribution and random initial phase
         if self.use_exponential:
             self.omega = 2 * np.pi * self.freq_lowest * np.power(self.freq_highest / self.freq_lowest, np.linspace(0, 1, self.rows))
         else:
             self.omega = 2 * np.pi * self.freq_lowest + 2 * np.pi * (self.freq_highest - self.freq_lowest) * \
                          np.linspace(0, 1, self.rows)
         self.phi0 = 2 * np.pi * np.random.rand(self.rows)
         self._init_waveform_cache_stereo()
         
    def _rnd(self):
        return np.random.rand()
    
    def _init_waveform_cache_stereo(self):
        # waveform cache
        tau1 = 0.5 / self.omega[-1]
        tau2 = 0.25 * tau1**2
        q, q2 = 0.0, 0.0
        yl, yr = 0.0, 0.0
        zl, zr = 0.0, 0.0

        for sample in range(self.sample_count):
            if self.use_bspline:
                q = 1.0 * (sample % self.samples_per_column) / (self.samples_per_column - 1)
                q2 = 0.5 * q**2

            j = sample // self.samples_per_column
            if j > self.columns - 1:
                j = self.columns - 1

            r = 1.0 * sample / (self.sample_count - 1)  # Binaural attenuation/delay parameter
            theta = (r - 0.5) * 2 * np.pi / 3
            x = 0.5 * self.acoustical_size_of_head_m * (theta + np.sin(theta))
            tl = sample * self.time_per_sample_s
            tr = tl

            if self.use_delay:
                tr += x / self.speed_of_sound_m_s  # Time delay model

            x = np.abs(x)
            sl, sr = 0.0, 0.0

            im1 = self._get_image_column(j - 1) if j > 0 else None
            im2 = self._get_image_column(j)
            im3 = self._get_image_column(j + 1) if j < self.columns - 1 else None

            for i in range(self.rows):
                a = self._calculate_a(im1, im2, im3, i, q, q2)
                sl += a * self.waveform_cache_left_channel[sample, i]
                sr += a * self.waveform_cache_right_channel[sample, i]

            if sample < self.sample_count // (5 * self.columns):
                sl = (2.0 * self._rnd() - 1.0) / self.scale  # Left "click"

            if tl < 0.0:
                sl = 0.0

            if tr < 0.0:
                sr = 0.0

            ypl, yl = yl, tau1 / self.time_per_sample_s + tau2 / (self.time_per_sample_s**2)
            yl = (sl + yl * ypl + tau2 / self.time_per_sample_s * zl) / (1.0 + yl)
            zl = (yl - ypl) / self.time_per_sample_s

            ypr, yr = yr, tau1 / self.time_per_sample_s + tau2 / (self.time_per_sample_s**2)
            yr = (sr + yr * ypr + tau2 / self.time_per_sample_s * zr) / (1.0 + yr)
            zr = (yr - ypr) / self.time_per_sample_s

            l = 0.5 + self.scale * 32768.0 * yl
            l = min(max(l, -32768), 32767)
            self.audio_data[sample, 0] = int(l)

            l = 0.5 + self.scale * 32768.0 * yr
            l = min(max(l, -32768), 32767)
            self.audio_data[sample, 1] = int(l)
            
            
    def _calculate_a(self, im1, im2, im3, i, q, q2):
        if self.use_bspline:
            if im1 is None or im3 is None:
                return im2[i]

            f1 = (q2 - q + 0.5)
            f2 = (0.5 + q - q**2)

            if im1 is None:
                return f1 * im2[i] + f2 * im3[i]
            elif im3 is None:
                return f1 * im1[i] + f2 * im2[i]
            else:
                return f1 * im1[i] + f2 * im2[i] + q2 * im3[i]
        else:
            return im2[i]
    def _get_image_column(self, index):
        if 0 <= index < self.columns:
            return self.image[:, index]
        return None

    def process(self, image):
        if not self.use_stereo:
            self.process_mono(image)
        else:
            self.process_stereo(image)

    def process_mono(self, image):
        raise NotImplementedError("Mono audio not implemented")

    def process_stereo(self, image):
        self.init_waveform_cache_stereo()

    def init_waveform_cache_stereo(self):
        self._init_waveform_cache_stereo()