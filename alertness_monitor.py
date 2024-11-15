import numpy as np
from pylsl import StreamInlet, resolve_stream
import time
from scipy import signal
from datetime import datetime
import os

class AlertnessMonitor:
    def __init__(self):
        print("Looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        self.inlet = StreamInlet(streams[0])
        
        self.info = self.inlet.info()
        self.fs = int(self.info.nominal_srate())
        self.channel_count = self.info.channel_count()
        
        ch = self.info.desc().child("channels").child("channel")
        self.channel_names = []
        for i in range(self.channel_count):
            self.channel_names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        
        if not self.channel_names:
            self.channel_names = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
        
        # Buffer for analysis (4 seconds of data)
        self.buffer_size = int(self.fs * 4)
        self.eeg_buffer = np.zeros((self.channel_count, self.buffer_size))
        
        self.freq_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
        
        self.window = signal.windows.hamming(self.buffer_size)
        self.nfft = int(max(256, 2**np.ceil(np.log2(self.buffer_size))))
        self.frequencies = np.fft.rfftfreq(self.nfft, d=1/self.fs)
        
        self.last_update_time = time.time()
        self.update_interval = 3
        self.state_history = []
        self.avg_window = 5

    def compute_band_powers(self, data):
        windowed_data = data * self.window
        fft_data = np.fft.rfft(windowed_data, n=self.nfft)
        power_spectrum = np.abs(fft_data) ** 2
        
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            freq_mask = (self.frequencies >= low_freq) & (self.frequencies <= high_freq)
            band_powers[band_name] = np.mean(power_spectrum[freq_mask])
        
        return band_powers

    def analyze_cognitive_state(self, band_powers):
        total_power = sum(band_powers.values())
        if total_power > 0:
            relative_powers = {band: power/total_power for band, power in band_powers.items()}
        else:
            relative_powers = {band: 0 for band in band_powers}
        
        # Calculate ratios
        theta_beta_ratio = band_powers['Theta'] / band_powers['Beta'] if band_powers['Beta'] > 0 else 0
        alpha_beta_ratio = band_powers['Alpha'] / band_powers['Beta'] if band_powers['Beta'] > 0 else 0
        
        # Determine cognitive state
        if theta_beta_ratio > 2.5:
            return "DROWSY - Take a break!"
        elif relative_powers['Delta'] > 0.4:
            return "MICROSLEEP RISK - Pull over!"
        elif relative_powers['Beta'] > 0.3 and relative_powers['Gamma'] > 0.15:
            return "HIGHLY ALERT"
        elif relative_powers['Beta'] > 0.25:
            return "ALERT"
        elif 0.6 < alpha_beta_ratio < 1.5:
            return "RELAXED"
        else:
            return "NORMAL"

    def update_buffer(self, new_data):
        if new_data.size > 0:
            new_samples = new_data.shape[0]
            self.eeg_buffer = np.roll(self.eeg_buffer, -new_samples, axis=1)
            self.eeg_buffer[:, -new_samples:] = new_data.T

    def get_smoothed_state(self, current_state):
        self.state_history.append(current_state)
        if len(self.state_history) > self.avg_window:
            self.state_history.pop(0)
        
        # Return most common state
        return max(set(self.state_history), key=self.state_history.count)

    def clear_terminal(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def run(self):
        print("Starting alertness monitoring...")
        print("Calibrating baseline... Please remain still for a few seconds...")
        time.sleep(5)
        self.clear_terminal()
        
        try:
            while True:
                chunk, timestamps = self.inlet.pull_chunk()
                
                if chunk:
                    chunk_data = np.array(chunk)
                    self.update_buffer(chunk_data)
                    
                    current_time = time.time()
                    if current_time - self.last_update_time >= self.update_interval:
                        # Average band powers across channels (excluding AUX)
                        avg_powers = {band: 0 for band in self.freq_bands}
                        for i in range(self.channel_count - 1):
                            channel_powers = self.compute_band_powers(self.eeg_buffer[i, :])
                            for band, power in channel_powers.items():
                                avg_powers[band] += power
                        
                        avg_powers = {band: power/(self.channel_count-1) for band, power in avg_powers.items()}
                        
                        # Get and smooth state
                        current_state = self.analyze_cognitive_state(avg_powers)
                        smoothed_state = self.get_smoothed_state(current_state)
                        
                        # Print the state with timestamp
                        current_time_str = datetime.now().strftime("%H:%M:%S")
                        
                        # Clear screen and print new state
                        self.clear_terminal()
                        print("\n" + "="*50)
                        print(f"Time: {current_time_str}")
                        print(f"Current State: {smoothed_state}")
                        print("="*50 + "\n")
                        
                        self.last_update_time = current_time
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping monitoring...")

if __name__ == "__main__":
    monitor = AlertnessMonitor()
    monitor.run()