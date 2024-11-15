import time
import pylsl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def collect_eeg_data(duration=60, chunk_size=12):
    """
    Collect EEG data for specified duration
    
    Args:
        duration (int): Duration to collect in seconds
        chunk_size (int): Number of samples to collect per chunk
    
    Returns:
        pd.DataFrame: Collected EEG data
    """
    # Find EEG stream
    print("Looking for EEG stream...")
    streams = pylsl.resolve_stream('type', 'EEG')
    
    # Wait for stream to become available
    timeout = 10
    start_wait = time.time()
    while not streams:
        if time.time() - start_wait > timeout:
            print("Timeout: No EEG stream found. Make sure start_stream.py is running.")
            return None
        print("Waiting for EEG stream...")
        time.sleep(1)
        streams = pylsl.resolve_stream('type', 'EEG')
    
    inlet = pylsl.StreamInlet(streams[0])
    
    # Get stream info
    info = inlet.info()
    channel_count = info.channel_count()
    
    # Get channel names
    ch = info.desc().child("channels").child("channel")
    channel_names = []
    for i in range(channel_count):
        channel_names.append(ch.child_value("label"))
        ch = ch.next_sibling()
    
    # If channel names are not available, use default names
    if not channel_names:
        channel_names = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
    
    # Initialize data collection
    eeg_data = []
    start_time = time.time()
    
    print(f"Starting data collection for {duration} seconds...")
    print(f"Collecting data from {len(channel_names)} channels: {channel_names}")
    
    try:
        while (time.time() - start_time) < duration:
            data, timestamp = inlet.pull_chunk(timeout=1.0, 
                                             max_samples=chunk_size)
            if data:
                # print(data)
                eeg_data.extend(data)
                plt.plot(data)
                plt.show()
                # Optional: Print a dot to show progress
                print(".", end="", flush=True)
                
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    
    print("\n")  # New line after progress dots
    
    # Convert to DataFrame
    if eeg_data:
        df = pd.DataFrame(eeg_data, columns=channel_names)
        print(f"Collected {len(df)} samples")
        return df
    else:
        print("No data collected")
        return None

def analyze_data(df):
    """
    Perform basic analysis on the EEG data
    """
    if df is None:
        return
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Calculate channel averages
    print("\nChannel Averages:")
    for channel in df.columns:
        print(f"{channel}: {df[channel].mean():.2f}")
        
    # Print data shape
    print(f"\nData shape: {df.shape} (rows x columns)")

def main():
    try:
        # Collect data
        data = collect_eeg_data(duration=30)  # Collect 30 seconds of data
        
        if data is not None:
            # Analyze data
            analyze_data(data)
            
            # Save data
            filename = f'muse_eeg_data_{int(time.time())}.csv'
            data.to_csv(filename, index=False)
            print(f"\nData saved to {filename}")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")

if __name__ == "__main__":
    main()