from muselsl import stream, list_muses

def main():
    device = ""

    muses = list_muses()
    for muse in muses:
        if muse['address'] == 'DD158D69-A103-9D19-3512-78E00C823F78':
            device = muse['address']
            print(muse, "Muse found")
            break
    
    if device == "":
        print("No device found")
        return -1
    
    stream(device)
    return 0

if __name__ == "__main__":
    main()

# Note: Streaming is synchronous, so code here will not execute until after the stream has been closed
print('Stream has ended')