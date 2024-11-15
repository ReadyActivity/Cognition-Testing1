import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream
import time
import pandas as pd
from datetime import datetime
import os

class EEGRecorder:
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
        
        # Recording state
        self.is_recording = False
        self.recorded_data = []
        self.recorded_timestamps = []
        self.should_quit = False
        
        # Create data directory if it doesn't exist
        self.data_dir = 'eeg_recordings'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.setup_visualization()
    
    def setup_visualization(self):
        # Create the figure and axes
        self.fig, self.axes = plt.subplots(self.channel_count, 1, 
                                         figsize=(15, 10), 
                                         sharex=True)
        if self.channel_count == 1:
            self.axes = [self.axes]
        
        # Set up the plots
        self.window_size = 5  # 5 seconds of data
        self.buffer_size = int(self.fs * self.window_size)
        self.buffers = [np.zeros(self.buffer_size) for _ in range(self.channel_count)]
        self.lines = []
        
        # Create a line for each channel
        t = np.linspace(-self.window_size, 0, self.buffer_size)
        for ax, name in zip(self.axes, self.channel_names):
            line, = ax.plot(t, np.zeros(self.buffer_size))
            self.lines.append(line)
            ax.set_ylabel(f'{name} (μV)')
            ax.grid(True)
            ax.set_ylim(-100, 100)
        
        self.axes[-1].set_xlabel('Time (s)')
        
        # Add recording status text
        self.status_text = self.fig.text(0.02, 0.98, 'Not Recording', 
                                       color='red',
                                       transform=self.fig.transFigure)
        
        # Connect the key press event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
    
    def on_key_press(self, event):
        if event.key == 'r':
            if self.is_recording:
                self.stop_recording()
            else:
                self.start_recording()
        elif event.key == 'q':
            self.should_quit = True
    
    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_data = []
            self.recorded_timestamps = []
            self.status_text.set_text('Recording...')
            self.status_text.set_color('green')
            print("\nRecording started...")
    
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.status_text.set_text('Not Recording')
            self.status_text.set_color('red')
            self.save_recording()
            print("\nRecording stopped and saved.")
    
    def save_recording(self):
        if not self.recorded_data:
            print("No data to save")
            return
        
        # Convert data to DataFrame
        data = np.array(self.recorded_data)
        timestamps = np.array(self.recorded_timestamps)
        
        df = pd.DataFrame(data, columns=self.channel_names)
        df['timestamp'] = timestamps
        
        # Generate filename with timestamp
        filename = os.path.join(
            self.data_dir,
            f'eeg_recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        
        # Print recording statistics
        duration = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
        print(f"Recording statistics:")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Samples: {len(df)}")
        print(f"Average sampling rate: {len(df)/duration:.2f} Hz")
    
    def update_plot(self):
        try:
            # Get chunk of data and timestamps
            chunk, timestamps = self.inlet.pull_chunk()
            
            if chunk:
                chunk = np.array(chunk)
                
                # Record data if recording is active
                if self.is_recording:
                    self.recorded_data.extend(chunk)
                    self.recorded_timestamps.extend(timestamps)
                
                for ch_ix in range(self.channel_count):
                    # Roll the buffer and add new data
                    self.buffers[ch_ix] = np.roll(self.buffers[ch_ix], -len(chunk))
                    self.buffers[ch_ix][-len(chunk):] = chunk[:, ch_ix]
                    
                    # Update line
                    self.lines[ch_ix].set_ydata(self.buffers[ch_ix])
                    
                    # Auto-scale y-axis
                    if np.ptp(self.buffers[ch_ix]) > 0:
                        margin = np.ptp(self.buffers[ch_ix]) * 0.1
                        self.axes[ch_ix].set_ylim(
                            np.min(self.buffers[ch_ix]) - margin,
                            np.max(self.buffers[ch_ix]) + margin
                        )
                
                plt.draw()
                plt.pause(0.01)
            
            return True
            
        except KeyboardInterrupt:
            return False
    
    def run(self):
        print("\nControls:")
        print("R - Start/Stop Recording")
        print("Q - Quit")
        print("\nStarting visualization...")
        print("Click on the plot window and use keyboard controls")
        
        plt.ion()  # Turn on interactive mode
        
        try:
            while not self.should_quit:
                self.update_plot()
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            if self.is_recording:
                self.stop_recording()
            plt.ioff()
            plt.close('all')

if __name__ == "__main__":
    recorder = EEGRecorder()
    recorder.run()