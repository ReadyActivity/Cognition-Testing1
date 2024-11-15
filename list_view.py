import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pylsl
import time

class EEGViewer:
    def __init__(self):
        # Find EEG stream
        print("Looking for EEG stream...")
        streams = pylsl.resolve_stream('type', 'EEG')
        
        while not streams:
            print("Waiting for EEG stream...")
            time.sleep(1)
            streams = pylsl.resolve_stream('type', 'EEG')
        
        self.inlet = pylsl.StreamInlet(streams[0])
        
        # Get stream info
        info = self.inlet.info()
        self.channel_count = info.channel_count()
        
        # Get channel names
        ch = info.desc().child("channels").child("channel")
        self.channel_names = []
        for i in range(self.channel_count):
            self.channel_names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        
        # If channel names are not available, use default names
        if not self.channel_names:
            self.channel_names = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
        
        # Initialize data buffer (5 seconds of data at 256Hz)
        self.buffer_size = 256 * 5
        self.data_buffer = np.zeros((self.buffer_size, self.channel_count))
        
        # Create figure and subplots
        self.fig, self.axes = plt.subplots(self.channel_count, 1, figsize=(12, 8), sharex=True)
        if self.channel_count == 1:
            self.axes = [self.axes]
        
        self.fig.suptitle('Live EEG Data')
        self.lines = []
        
        # Initialize plots
        for i, ax in enumerate(self.axes):
            line, = ax.plot(np.zeros(self.buffer_size))
            self.lines.append(line)
            ax.set_ylabel(f'{self.channel_names[i]} (μV)')
            ax.grid(True)
        
        self.axes[-1].set_xlabel('Samples')
        
        # Add text for signal quality indicators
        self.quality_text = self.fig.text(0.02, 0.98, '', transform=self.fig.transFigure, 
                                        verticalalignment='top')
    
    def update(self, frame):
        # Get chunk of data
        data, timestamps = self.inlet.pull_chunk(timeout=0.0, 
                                               max_samples=self.buffer_size)
        
        if data:
            # Update buffer with new data
            new_data = np.array(data)
            samples_to_add = len(new_data)
            
            if samples_to_add > 0:
                # Roll buffer and add new data
                self.data_buffer = np.roll(self.data_buffer, -samples_to_add, axis=0)
                self.data_buffer[-samples_to_add:, :] = new_data
                
                # Update plots
                for i, line in enumerate(self.lines):
                    line.set_ydata(self.data_buffer[:, i])
                    
                    # Auto-scale y-axis occasionally
                    if frame % 30 == 0:
                        self.axes[i].relim()
                        self.axes[i].autoscale_view()
        
        # Update signal quality indicator (basic threshold-based)
        quality_msg = "Signal Quality:\n"
        for i, ch_name in enumerate(self.channel_names[:-1]):  # Exclude AUX channel
            # Simple signal quality check based on amplitude
            recent_data = self.data_buffer[-256:, i]  # Last second of data
            if np.std(recent_data) < 1:
                quality = "Poor"
            elif np.std(recent_data) < 10:
                quality = "Good"
            else:
                quality = "Excellent"
            quality_msg += f"{ch_name}: {quality}\n"
        
        self.quality_text.set_text(quality_msg)
        
        return self.lines + [self.quality_text]
    
    def run(self):
        # Set up animation
        self.anim = FuncAnimation(self.fig, self.update, interval=100, 
                                blit=True)
        
        # Configure plot appearance
        plt.tight_layout()
        self.fig.subplots_adjust(right=0.85)  # Make room for signal quality
        
        # Show plot
        plt.show()

if __name__ == "__main__":
    viewer = EEGViewer()
    viewer.run()