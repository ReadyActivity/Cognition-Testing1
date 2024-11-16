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
        
        # Calculate Alpha/Theta Ratio -- higher ratio: more alert
        if 'Alpha' in band_powers and 'Theta' in band_powers:
            ratio = band_powers['Alpha'] / band_powers['Theta']
            band_powers['Alpha/Theta Ratio'] = ratio

        # Calculate Alpha/Beta Ratio -- higher ratio: more relaxed
        if 'Alpha' in band_powers and 'Beta' in band_powers:
            ratio = band_powers['Alpha'] / band_powers['Beta']
            band_powers['Alpha/Beta Ratio'] = ratio
        
        # Calculate Theta/Beta Ratio -- higher ratio: more distracted, lower ratio: more focused
        if 'Theta' in band_powers and 'Beta' in band_powers:
            ratio = band_powers['Theta'] / band_powers['Beta']
            band_powers['Theta/Beta Ratio'] = ratio
        
        # Calculate Delta/Theta Ratio -- higher ratio: more sleepy
        if 'Delta' in band_powers and 'Theta' in band_powers:
            ratio = band_powers['Delta'] / band_powers['Theta']
            band_powers['Delta/Theta Ratio'] = ratio
        
        # Calculate Gamma/Alpha Ratio -- higher ratio: more 
        if 'Gamma' in band_powers and 'Alpha' in band_powers:
            ratio = band_powers['Gamma'] / band_powers['Alpha']
            band_powers['Gamma/Alpha Ratio'] = ratio
        
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
                        # Determine brain state
                        self.determine_brain_state(results)
                        
                        # Here you can add your own analysis functions
                        # self.your_custom_analysis(results)
                
                time.sleep(0.5)  # Small pause to prevent overwhelming the CPU
                
        except KeyboardInterrupt:
            print("\nStopping analysis...")

    def determine_brain_state(self, results):
        # Initialize brain state with descriptive levels
        brain_state = {
            "alertness_level": "",
            "stress_level": "",
            "fatigue_level": "",
            "focus_level": ""
        }

        # Extract ratios for specific channels
        band_powers_af7 = results['channels']['AF7']['band_powers']
        band_powers_af8 = results['channels']['AF8']['band_powers']

        # Compute average ratios across channels
        ratios = {}
        ratio_keys = [
            'Alpha/Theta Ratio',  # Alertness
            'Alpha/Beta Ratio',   # Relaxation/Stress
            'Theta/Beta Ratio',   # Focus/Distraction
            'Delta/Theta Ratio',  # Sleepiness
            'Gamma/Alpha Ratio'   # Cognitive activity
        ]
        
        for key in ratio_keys:
            ratios[key] = (
                band_powers_af7.get(key, 0) + band_powers_af8.get(key, 0)
            ) / 2

        # Determine alertness level (Alpha/Theta Ratio)
        if ratios['Alpha/Theta Ratio'] < 1.5:
            brain_state['alertness_level'] = "Relaxed"
        elif 1.5 <= ratios['Alpha/Theta Ratio'] <= 2.5:
            brain_state['alertness_level'] = "Moderately Alert"
        else:
            brain_state['alertness_level'] = "Highly Alert"

        # Determine stress level (Alpha/Beta Ratio)
        if ratios['Alpha/Beta Ratio'] > 0.3:
            brain_state['stress_level'] = "Low Stress"
        elif 0.1 <= ratios['Alpha/Beta Ratio'] <= 0.3:
            brain_state['stress_level'] = "Moderate Stress"
        else:
            brain_state['stress_level'] = "High Stress"

        # Determine fatigue level (Delta/Theta Ratio)
        if ratios['Delta/Theta Ratio'] > 10:
            brain_state['fatigue_level'] = "Sleepy"
        elif 4 <= ratios['Delta/Theta Ratio'] <= 10:
            brain_state['fatigue_level'] = "Tired"
        else:
            brain_state['fatigue_level'] = "Awake"

        # Determine focus level (Theta/Beta Ratio)
        if ratios['Theta/Beta Ratio'] > 2:
            brain_state['focus_level'] = "Distracted"
        elif 1 <= ratios['Theta/Beta Ratio'] <= 2:
            brain_state['focus_level'] = "Moderately Focused"
        else:
            brain_state['focus_level'] = "Highly Focused"

        # Output the results
        print("Brain State:\n")
        for key, value in brain_state.items():
            print(f"{key.capitalize()}: {value}")

        return brain_state

if __name__ == "__main__":
    analyzer = EEGAnalyzer()
    analyzer.run()