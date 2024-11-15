import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_stream
import time

def main():
    # First resolve an EEG stream on the lab network
    print("Looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    
    # Get the stream info
    info = inlet.info()
    fs = int(info.nominal_srate())
    channel_count = info.channel_count()
    
    # Get channel names
    ch = info.desc().child("channels").child("channel")
    channel_names = []
    for i in range(channel_count):
        channel_names.append(ch.child_value("label"))
        ch = ch.next_sibling()
    
    if not channel_names:
        channel_names = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
    
    # Create the figure and axes
    fig, axes = plt.subplots(channel_count, 1, figsize=(15, 10), sharex=True)
    if channel_count == 1:
        axes = [axes]
    
    # Set up the plots
    window_size = 5  # 5 seconds of data
    buffer_size = int(fs * window_size)
    buffers = [np.zeros(buffer_size) for _ in range(channel_count)]
    lines = []
    
    # Create a line for each channel
    t = np.linspace(-window_size, 0, buffer_size)
    for ax, name in zip(axes, channel_names):
        line, = ax.plot(t, np.zeros(buffer_size))
        lines.append(line)
        ax.set_ylabel(f'{name} (μV)')
        ax.grid(True)
        ax.set_ylim(-100, 100)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    # Update function for the plot
    def update_plot():
        try:
            # Get chunk of data and timestamps
            chunk, timestamps = inlet.pull_chunk()
            
            if chunk:
                chunk = np.array(chunk)
                for ch_ix in range(channel_count):
                    # Roll the buffer and add new data
                    buffers[ch_ix] = np.roll(buffers[ch_ix], -len(chunk))
                    buffers[ch_ix][-len(chunk):] = chunk[:, ch_ix]
                    
                    # Update line
                    lines[ch_ix].set_ydata(buffers[ch_ix])
                    
                    # Auto-scale y-axis (optional)
                    if np.ptp(buffers[ch_ix]) > 0:
                        margin = np.ptp(buffers[ch_ix]) * 0.1
                        axes[ch_ix].set_ylim(
                            np.min(buffers[ch_ix]) - margin,
                            np.max(buffers[ch_ix]) + margin
                        )
                
                plt.draw()
                plt.pause(0.01)  # Small pause to allow the plot to update
                
            return True
        
        except KeyboardInterrupt:
            return False
    
    print("Starting data visualization... Press Ctrl+C to stop")
    plt.ion()  # Turn on interactive mode
    
    try:
        while update_plot():
            pass
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    finally:
        plt.ioff()
        plt.close('all')

if __name__ == "__main__":
    main()