import subprocess
import threading
import wave
import numpy as np
import os
import winsound
import time

class AudioData:
    audio_mutex = threading.Lock()
    
    def __init__(self, card_number: int, sample_freq_Hz: int, sample_count: int, use_stereo: bool):
        self.card_number = card_number
        self.sample_count = sample_count
        self.sample_freq_Hz = sample_freq_Hz
        self.use_stereo = use_stereo
        self.verbose = False
        self.volume = -1
        self.new_volume = -1
        
    def save_to_wav_file(self, audio_data, filename: str):
        bytes_per_sample = 4 if self.use_stereo else 2
        audio_data = np.array(audio_data)
        
        with wave.open(filename, 'wb') as fp:
            fp.setnchannels(2 if self.use_stereo else 1)
            fp.setsampwidth(bytes_per_sample)
            fp.setframerate(self.sample_freq_Hz)
            fp.setnframes(self.sample_count)
            fp.writeframes(audio_data.tobytes())
            
    def play(self):
        self.update_volume()

        bytes_per_sample = 4 if self.use_stereo else 2

        if os.name == 'posix':  # Linux or macOS
            cmd = [
                "aplay",
                "--nonblock",
                f"-r{self.sample_freq_Hz}",
                f"-c{'2' if self.use_stereo else '1'}",
                "-fS16_LE",
                f"-D plughw:{self.card_number}"
            ]
            if not self.verbose:
                cmd.append("-q")
            else:
                print(" ".join(cmd))

            with AudioData.audio_mutex:
                subprocess.run(cmd, input=self.sample_buffer.tobytes(), check=True)

        elif os.name == 'nt':  # Windows
            temp_wav_filename = "temp.wav"
            self.save_to_wav_file(temp_wav_filename)
            winsound.PlaySound(temp_wav_filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
            time.sleep(1)  # Adjust the duration based on your needs
            os.remove(temp_wav_filename)
        else:
            print("Unsupported operating system")
            return

                
    def play_wav(self, filename):
        cmd = f"apply {filename} -D hw:{self.card_number}"
        
        with AudioData.audio_mutex:
            status = subprocess.call(cmd, shell=True)
            
    def set_volume(self, new_volume):
        self.new_volume = new_volume
        
    def update_volume(self):
        if self.new_volume == self.new_volume or self.new_volume == -1:
            return 0
        self.volume = self.new_volume
        
        cmd = f"amixer -c {self.card_number} controls | grep MIXER | grep Playback | grep Volume | sed s/[^0-9]*//g"
        with AudioData.audio_mutex:
            try:
                res = subprocess.run(cmd, shell= True, stdout= subprocess.PIPE, text= True, check=True)
                numid = int(res.stdout.strip())
                cmd = f"amixer -c {self.card_number} cset numid={numid} {self.newvolume}% -q"
                subprocess.run(cmd, shell= True, check=True)
            except subprocess.CalledProcessError:
                return -1
        
        return 0
    def speak(self, text):
        self.update_volume()
        cmd =  f"espeak --stdout \"{text}\" | aplay -q -D plughw:{self.card_number}"
        with AudioData.audio_mutex:
            res = subprocess.run(cmd, shell=True)
        return res.returncode == 0
    