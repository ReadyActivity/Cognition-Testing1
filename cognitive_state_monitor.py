import numpy as np
from pylsl import StreamInlet, resolve_stream
import time
from scipy import signal
from datetime import datetime

class CognitiveStateMonitor:
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
        
        # Add a flag to track if we're connected
        self.is_connected = True
        
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
        
        # Return a dictionary with state information
        state_info = {
            'state': 'NORMAL',
            'alert_level': 2,  # 1: Danger, 2: Normal, 3: Good
            'details': {}
        }
        
        # Store the ratios and powers for potential use in the app
        state_info['details'] = {
            'theta_beta_ratio': theta_beta_ratio,
            'alpha_beta_ratio': alpha_beta_ratio,
            'relative_powers': relative_powers
        }
        
        # Determine cognitive state
        if theta_beta_ratio > 2.5:
            state_info['state'] = 'DROWSY'
            state_info['alert_level'] = 1
        elif relative_powers['Delta'] > 0.4:
            state_info['state'] = 'MICROSLEEP'
            state_info['alert_level'] = 1
        elif relative_powers['Beta'] > 0.3 and relative_powers['Gamma'] > 0.15:
            state_info['state'] = 'HIGHLY_ALERT'
            state_info['alert_level'] = 3
        elif relative_powers['Beta'] > 0.25:
            state_info['state'] = 'ALERT'
            state_info['alert_level'] = 3
        elif 0.6 < alpha_beta_ratio < 1.5:
            state_info['state'] = 'RELAXED'
            state_info['alert_level'] = 2
        
        return state_info

    def update_buffer(self, new_data):
        if new_data.size > 0:
            new_samples = new_data.shape[0]
            self.eeg_buffer = np.roll(self.eeg_buffer, -new_samples, axis=1)
            self.eeg_buffer[:, -new_samples:] = new_data.T

    def get_smoothed_state(self, current_state):
        self.state_history.append(current_state['state'])
        if len(self.state_history) > self.avg_window:
            self.state_history.pop(0)
        
        # Get most common state
        most_common_state = max(set(self.state_history), key=self.state_history.count)
        
        # Update the current state with the smoothed state
        current_state['state'] = most_common_state
        return current_state

    def get_current_state(self):
        """
        Main method to be called from Android Studio.
        Returns the current cognitive state or None if there's an error.
        """
        try:
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
                    
                    self.last_update_time = current_time
                    return smoothed_state
            
            return None  # No new data available
            
        except Exception as e:
            self.is_connected = False
            return {
                'state': 'ERROR',
                'alert_level': 1,
                'details': {'error': str(e)}
            }
    
    def is_device_connected(self):
        """Check if the device is still connected"""
        return self.is_connected
    
    def disconnect(self):
        """Clean up resources"""
        self.is_connected = False
        self.inlet = None

# Example usage in Python (for testing)
if __name__ == "__main__":
    monitor = CognitiveStateMonitor()
    try:
        while True:
            state = monitor.get_current_state()
            if state:
                print(f"State: {state['state']}, Alert Level: {state['alert_level']}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        monitor.disconnect()