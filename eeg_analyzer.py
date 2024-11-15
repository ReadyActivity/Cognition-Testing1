import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream
import time
from scipy import signal
from datetime import datetime
import os

class EEGAnalyzer:
    def __init__(self):
        # First resolve an EEG stream on the lab network
        print("Looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        self.inlet = StreamInlet(streams[0])
        
        # Get the stream info
        self.info = self.inlet.info()
        self.fs = int(self.info.nominal_srate())
        self.channel_count = self.info.channel_count()
        
        # Get channel names
        ch = self.info.desc().child("channels").child("channel")
        self.channel_names = []
        for i in range(self.channel_count):
            self.channel_names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        
        if not self.channel_names:
            self.channel_names = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
        
        # Buffer for frequency analysis (2 seconds of data)
        self.buffer_size = int(self.fs * 2)  # 2 seconds buffer
        self.eeg_buffer = np.zeros((self.channel_count, self.buffer_size))
        
        # Define frequency bands (in Hz)
        self.freq_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
    
    def update_buffer(self, new_data):
        """Update the buffer with new data"""
        if new_data.size > 0:
            new_samples = new_data.shape[0]
            # Roll the buffer to make room for new data
            self.eeg_buffer = np.roll(self.eeg_buffer, -new_samples, axis=1)
            # Add new data
            self.eeg_buffer[:, -new_samples:] = new_data.T
    
    def compute_band_powers(self, channel_data):
        """Compute the power in each frequency band for a single channel"""
        # Apply Hanning window
        windowed_data = signal.windows.hann(len(channel_data)) * channel_data
        
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(windowed_data, self.fs, nperseg=self.fs)
        
        # Calculate absolute power in each band
        band_powers = {}
        for band_name, (low, high) in self.freq_bands.items():
            # Find frequencies that fall within the band
            mask = (freqs >= low) & (freqs <= high)
            # Calculate mean power in band
            band_powers[band_name] = np.mean(psd[mask])
        
        return band_powers
    
    def analyze_eeg(self):
        """Analyze the current EEG buffer"""
        results = {
            'timestamp': datetime.now(),
            'channels': {}
        }
        
        # Analyze each channel
        for i, channel_name in enumerate(self.channel_names[:-1]):  # Exclude AUX channel
            # Get data for this channel
            channel_data = self.eeg_buffer[i, :]
            
            # Skip if data is all zeros
            if not np.any(channel_data):
                continue
            
            # Compute band powers
            band_powers = self.compute_band_powers(channel_data)
            
            # Store results
            results['channels'][channel_name] = {
                'band_powers': band_powers,
                'mean': np.mean(channel_data),
                'std': np.std(channel_data),
                'min': np.min(channel_data),
                'max': np.max(channel_data)
            }
        
        return results
    
    def process_results(self, results):
        """Process and print analysis results"""
        print(f"\nTimestamp: {results['timestamp']}")
        print("\nBand Powers (μV²):")
        
        # Print band powers for each channel
        for channel, data in results['channels'].items():
            print(f"\n{channel}:")
            for band, power in data['band_powers'].items():
                print(f"  {band}: {power:.2f}")
            
            # Print some basic statistics
            print(f"  Mean: {data['mean']:.2f} μV")
            print(f"  Std: {data['std']:.2f} μV")
    
    def run(self):
        print(f"Starting EEG analysis...")
        print(f"Sampling rate: {self.fs} Hz")
        print(f"Analyzing channels: {', '.join(self.channel_names[:-1])}")  # Exclude AUX
        print(f"Buffer size: {self.buffer_size} samples ({self.buffer_size/self.fs:.1f} seconds)")
        
        try:
            while True:
                # Get chunk of data
                chunk, timestamps = self.inlet.pull_chunk()
                
                if chunk:
                    # Convert to numpy array
                    chunk_data = np.array(chunk)
                    
                    # Update buffer with new data
                    self.update_buffer(chunk_data)
                    
                    # Perform analysis when we have a full buffer
                    if np.any(self.eeg_buffer):
                        results = self.analyze_eeg()
                        self.process_results(results)
                        
                        # Here you can add your own analysis functions
                        # self.your_custom_analysis(results)
                
                time.sleep(0.1)  # Small pause to prevent overwhelming the CPU
                
        except KeyboardInterrupt:
            print("\nStopping analysis...")

def your_custom_analysis(eeg_data):
    """
    Add your own analysis functions here.
    This is just an example template.
    """
    # Example: Calculate alpha/theta ratio
    # alpha_power = ...
    # theta_power = ...
    # ratio = alpha_power / theta_power
    pass

if __name__ == "__main__":
    analyzer = EEGAnalyzer()
    analyzer.run()