import numpy as np
from AudioData import AudioData
from scipy.io.wavfile import write
from scipy.interpolate import RectBivariateSpline
from scipy import signal
import math

class ImageToSoundscapeConverter:
    def __init__(self, rows: int, columns: int, freq_lowest = 500, freq_highest = 5000, sample_freq_Hz = 44100,
                 total_time_sec = 1.05, use_exponential = True, use_stereo = True, use_delay = True, use_fade = True,
                 use_diffraction = True, use_bspline = True, speed_of_sound_ms = 340, acoustical_size_of_head_m = 0.2, HIFI = True):
        self.rows = rows
        self.columns = columns
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
        self.audio_data = AudioData(card_number=self.card_number, sample_freq_Hz=self.sample_freq_Hz, 
                                    sample_count=self.sample_count, use_stereo=self.use_stereo)
        self.HIFI = HIFI
        if self.HIFI:
            self.sso = 0
            self.ssm = 32768
        else:
            self.sso = 128
            self.ssm = 128
        
        self.sample_counts = np.arange(self.sample_count) # arrays of sample count
        self.waveform_cache_left_channel = np.zeros((self.sample_count *rows))
        self.waveform_cache_right_channel = np.zeros((self.sample_count *rows))
        self.omege, self.phi0 = self.initialize_omega_phi0()
        self.tau1 = 0.5 / self.omega[rows-1]
        self.tau2 = 0.25 * (self.tau1**2)
        self.TwoPi = 2 * np.pi
        self.stereo_left = []
        self.stereo_right = []
        
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
    
    def process_image(self, image_array):
        # Check for image size matches with the class object
        if isinstance(image_array, np.ndarray):
            raise ValueError("Image must be in array format")
        if image_array.shape[0] == self.rows or image_array.shape[1] == self.columns:
            raise ValueError(f"Input image size must{image_array.shape} match with class object {(self.rows, self.columns)}")
        k = 0
        b = 0
        stereo_right = []
        stereo_left = []
        mono_audio = []
        
        while k < self.sample_count:
                if use_bspline:
                    q, q2 = self.calculate_q(k)
                    
                j = int(k / self.sample_per_colums)
                j = columns-1 if j > columns-1 else j        
                
                if self.use_stereo:
                    tl = tr = k * self.time_per_sample_s
                    r = 1.0 * k / (self.sample_count - 1)  # Binaural attenuation/delay parameter
                    theta = (r - 0.5) * self.TwoPi / 3
                    x = 0.5 * self.freq_highest * (theta + math.sin(theta))
                    x = np.abs(x)
                    sl = sr = 0.0
                    hrtfl = hrtfr = 1.0
                else:
                    t = k * self.time_per_sample_s
                if use_delay and self.use_stereo:
                    tr = tr + x / self.speed_of_sound_ms  # Time delay model
                
                for i in range(0, self.rows):
                    if self.use_diffraction and self.use_stereo:
                        # First order frequency-dependent azimuth diffraction model
                        hrtf = 1.0 if (self.TwoPi*self.speed_of_sound_ms/self.omega[i] > x) else self.TwoPi* self.speed_of_sound_ms/(x* self.omega[i])
                        if theta < 0.0:
                            hrtfl =  1.0
                            hrtfr = hrtf
                        else:
                            hrtfl = hrtf
                            hrtfr =  1.0
                    if use_fade and self.use_stereo:
                        # Simple frequency-independent relative fade model
                        hrtfl *= (1.0 - 0.7 * r)
                        hrtfr *= (0.3 + 0.7 * r)
                        
                    if use_bspline:
                        pixel_row = image_array[i]
                        pixel = self.bspline_interpolation(column= j, q = q, q2= q2, pixel = pixel_row)
                    else:
                        pixel = image_array[i][j]
                        
                    if self.use_stereo:
                        sl = sl + hrtfl * pixel * math.sin(self.omega[i] * tl + self.phi0[i])
                        sr = sr + hrtfr * pixel * math.sin(self.omega[i] * tr + self.phi0[i])
                        sl = (2.0 * self.random_no() - 1.0) / self.scale if (k < self.sample_count / (5 * columns)) else sl  # Left "click"
                        if tl < 0.0:
                            sl = 0.0
                        if tr < 0.0:
                            sr = 0.0
                        ypl = yl
                        yl = self.tau1 / self.time_per_sample_s + self.tau2 / (self.time_per_sample_s * self.time_per_sample_s)
                        yl = (sl + yl * ypl + self.tau2 / self.time_per_sample_s * zl) / (1.0 + yl)
                        zl = (yl - ypl) / self.time_per_sample_s
                        ypr = yr
                        yr = self.tau1 / self.time_per_sample_s + self.tau2 / (self.time_per_sample_s * self.time_per_sample_s)
                        yr = (sr + yr * ypr + self.tau2 / self.time_per_sample_s * zr) / (1.0 + yr)
                        zr = (yr - ypr) / self.time_per_sample_s
                        
                        left_channel = self.sso + 0.5 + self.scale * self.ssm * yl
                        if left_channel >= self.sso-1 + self.ssm:
                            left_channel = self.sso-1 + self.ssm
                        if left_channel < self.sso - self.ssm:
                            left_channel = self.sso - self.ssm
                    
                        # Left channel
                        stereo_left.append(left_channel)
                        
                        right_channel = self.sso + 0.5 + self.scale * self.ssm * yr
                        if right_channel >= self.sso-1 + self.ssm:
                            right_channel = self.sso-1 + self.ssm 
                        if right_channel < self.sso - self.ssm:
                            right_channel = self.sso - self.ssm
                        stereo_right.append(right_channel)
                    else:
                        s += pixel * math.sin(self.omega[i] * t + self.phi0[i])
                        yp = y
                        y = self.tau1 / self.time_per_sample_s + self.tau2 / (self.time_per_sample_s * self.time_per_sample_s)
                        y = (s + y * yp + self.tau2 / self.time_per_sample_s * z) / (1.0 + y)
                        z = (y - yp) / self.time_per_sample_s
                        channel = self.sso + 0.5 + self.scale * self.ssm * y  # y = 2nd order filtered s
                        if channel >= self.sso-1 + self.ssm:
                            channel = self.sso-1 + self.ssm
                        if channel < self.sso - self.ssm:
                            channel = self.sso - self.ssm
                        mono_audio.append(channel)
                    k = k + 1
                    
                return stereo_left, stereo_right, mono_audio
                
            
    def calculate_q(self, k):
        q = 1.0 * (k % self.sample_per_colums) / (self.sample_per_colums - 1)
        q2 = 0.5 * q * q
        return q, q2
    
    def bspline_interpolation(self, column, q, q2, pixel):
        if column == 0:
            return  (1.0 - q2) * pixel[column] + q2 * pixel[column + 1]
        elif column == self.columns -1:
            return (q2 - q + 0.5) * pixel[column-1] + (0.5 + q - q2) * pixel[column]
        else:
            return (q2 - q + 0.5) * pixel[column-1] + (0.5 + q - q * q) * pixel[column] + q2 * pixel[column+1]
    
    def fade(self, hrtfl, hrtfr, r):
        # Simple frequency-independent relative fade model
        hrtfl *= (1.0 - 0.7 * r)
        hrtfr *= (0.3 + 0.7 * r)
    
    def filter_wave(self, wave, t_values, second_order = True, azimuthal = False):
        # Filter parameters
        omega_n = 2 * np.pi * 2  # Natural frequency
        zeta = 0.7  # Damping ratio
        # Second-order filter transfer function
        if second_order:
            numerator = [omega_n**2]
            denominator = [1, 2 * zeta * omega_n, omega_n**2]
        elif azimuthal:
            # Frequency-dependent azimuth diffraction model
            T = 0.1  # Time constant
            numerator = [1]
            denominator = [1, T]
        else:
            numerator = 1
            denominator = 1
        sys = signal.TransferFunction(numerator, denominator)
        # Apply the filter to the sine wave
        t, filter_wave, _ = signal.lsim(sys, wave, t_values)
        
        return filter_wave

    def fade_wave(self, wave, fade_factor = 0.2):
        # Apply the fading model
        wave = wave * fade_factor
        return wave

        


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