import numpy as np
from AudioData import AudioData
from scipy.io.wavfile import write
from scipy.interpolate import RectBivariateSpline
from scipy import signal

class ImageToSoundscapeConverter:
    def __init__(self, image_array: np.array(), freq_lowest = 500, freq_highest = 5000, sample_freq_Hz = 44100,
                 total_time_sec = 1.05, use_exponential = True, use_stereo = True, use_delay = True, use_fade = True,
                 use_diffraction = True, use_bspline = True, speed_of_sound_ms = 340, acoustical_size_of_head_m = 0.2):
        self.image = image_array
        self.rows = image_array.shape[0]
        self.columns = image_array.shape[1]
        self.freq_highest = freq_highest # highest frequency
        self.freq_lowest = freq_lowest # lowest frequency
        self.sample_freq_Hz = sample_freq_Hz
        self.total_time_sec = total_time_sec
        self.use_exponential = use_exponential # using linear or exponential distributions of frequency
        self.use_stereo = use_stereo 
        self.use_delay = use_delay 
        self.use_fade = use_fade
        self.use_diffraction = use_diffraction # First order frequency-dependent azimuth diffraction model
        self.use_bspline = use_bspline 
        self.speed_of_sound_ms = speed_of_sound_ms
        self.acoustical_size_of_head_m  = acoustical_size_of_head_m
        
        
        self.sample_count = 2 * int(0.5 * sample_freq_Hz * total_time_sec)
        self.sample_per_column = int(self.sample_count/columns)
        self.scale = 0.5 / np.sqrt(rows)
        self.time_per_sample_s = 1.0 / sample_freq_Hz  # Initialize time_per_sample_s 
        self.card_number = 0
        self.audio_data = AudioData(card_number=self.card_number, sample_freq_Hz=self.sample_freq_Hz, sample_count=self.sample_count, use_stereo=self.use_stereo)
        self.sample_counts = np.arange(self.sample_count) # arrays of sample count
        self.waveform_cache_left_channel = np.zeros((self.sample_count *rows))
        self.waveform_cache_right_channel = np.zeros((self.sample_count *rows))
        self.omege, self.phi0 = self.initialize_omega_phi0()
        
    def initialize_omega_phi0(self):
         '''
         initialize omega and phi0 values accrding to highest and lowest frequencies
         '''
         if self.use_exponential:
             omega = 2 * np.pi * self.freq_lowest * np.power(self.freq_highest / self.freq_lowest, np.linspace(0, 1, self.rows))
         else:
             omega = 2 * np.pi * self.freq_lowest + 2 * np.pi * (self.freq_highest - self.freq_lowest) * \
                          np.linspace(0, 1, self.rows)
         phi0 = 2 * np.pi * np.random.rand(self.rows)
         return omega, phi0
    
    def random_no(self):
        '''
        generates random number between 0 and 1
        '''
        return round(np.random.rand(), 2)
    
    def _init_waveform_cache_stereo(self):
        # waveform cache
        tau1 = 0.5 / self.omega[-1]
        tau2 = 0.25 * tau1**2
        q, q2 = 0.0, 0.0
        yl, yr = 0.0, 0.0
        zl, zr = 0.0, 0.0

        for sample in range(self.sample_count):
            if self.use_bspline:
                q = 1.0 * (sample % self.sample_per_column) / (self.sample_per_column - 1)
                q2 = 0.5 * q**2

            j = sample // self.sample_per_column
            if j > self.columns - 1:
                j = self.columns - 1

            r = 1.0 * sample / (self.sample_count - 1)  # Binaural attenuation/delay parameter
            theta = (r - 0.5) * 2 * np.pi / 3
            x = 0.5 * self.acoustical_size_of_head_m * (theta + np.sin(theta))
            tl = sample * self.time_per_sample_s
            tr = tl

            if self.use_delay:
                tr += x / self.speed_of_sound_ms  # Time delay model

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
            
            
    def bspline_interpolation(self):
        # Generate a finer grid for interpolation
        new_rows, new_cols = self.rows * 2, self.columns * 2
        new_x = np.linspace(0, self.columns - 1, new_cols)
        new_y = np.linspace(0, self.rows - 1, new_rows)
        
        # Create B-spline interpolator
        spline = RectBivariateSpline(range(self.rows), range(self.columns), self.image)
        # Evaluate the B-spline at the new points
        interpolated_image = spline(new_y, new_x)
        
        return interpolated_image
    
    def second_order_filter(self, wave, t_values):
        # Filter parameters
        omega_n = 2 * np.pi * 2  # Natural frequency
        zeta = 0.7  # Damping ratio
        # Second-order filter transfer function
        numerator = [omega_n**2]
        denominator = [1, 2 * zeta * omega_n, omega_n**2]
        sys = signal.TransferFunction(numerator, denominator)
        # Apply the filter to the sine wave
        t, filter_wave, _ = signal.lsim(sys, wave, t_values)
        
        return filter_wave
        
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
        


rows = 10
columns = 10
freq_lowest = 100
freq_highest = 1000
sample_freq_Hz = 44100
total_time_s = 5
use_exponential = True
use_stereo = True
use_delay = True
use_fade = True
use_diffraction = True
use_bspline = True
speed_of_sound_ms = 343  # Speed of sound in air at 20 degrees Celsius
acoustical_size_of_head_m = 0.2
image = np.random.rand(rows, columns) * 255 

converter = ImageToSoundscapeConverter(image, freq_lowest, freq_highest, sample_freq_Hz,
                                        total_time_s, use_exponential, use_stereo, use_delay,
                                        use_fade, use_diffraction, use_bspline, speed_of_sound_ms,
                                        acoustical_size_of_head_m)

# converter.process(image)
print(converter._rnd())
print(converter.rnd())