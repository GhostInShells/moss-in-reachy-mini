import pyaudio

pa = pyaudio.PyAudio()


def safe(fn, label):
    try:
        return fn()
    except Exception as e:
        print(label, "ERROR:", repr(e))
        return None


print("device_count =", pa.get_device_count())
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    print(
        i,
        info.get("name"),
        "in",
        info.get("maxInputChannels"),
        "out",
        info.get("maxOutputChannels"),
        "rate",
        info.get("defaultSampleRate"),
    )

safe(pa.get_default_input_device_info, "default_input")
safe(pa.get_default_output_device_info, "default_output")


def try_open(dev, rate):
    kwargs = {"format": pyaudio.paInt16, "channels": 1, "rate": rate, "input": True, "frames_per_buffer": 1024}
    if dev is not None:
        kwargs["input_device_index"] = dev
    s = pa.open(**kwargs)
    s.close()


for dev in [1, 0, None]:
    for rate in [48000, 44100, 16000]:
        try:
            try_open(dev, rate)
            print("OPEN OK dev=", dev, "rate=", rate)
            break
        except Exception as e:
            print("OPEN FAIL dev=", dev, "rate=", rate, "err=", repr(e))

pa.terminate()
