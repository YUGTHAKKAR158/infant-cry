"""
Step 1 of the audio pipeline: get a raw audio file into memory as a waveform.

A "waveform" here just means a 1-D NumPy array of amplitude values, one per
sample, plus the sample rate (samples per second) that tells us how to
interpret time from that array. Everything downstream (filtering, feature
extraction) operates on this array.

We use `soundfile` here instead of Python's built-in `wave` module because the
raw dataset mixes PCM16 WAV, float WAV, and Ogg/Vorbis files (see
docs/dataset.md) - `wave` can only read plain PCM WAV and fails on the rest.
`soundfile` (backed by libsndfile) reads all of these transparently.
"""
import numpy as np
import soundfile as sf


def load_audio_av(file_path):
    import av
    container = av.open(file_path)
    stream = container.streams.audio[0]
    sample_rate = stream.rate or 16000
    frames = []
    for frame in container.decode(stream):
        # Convert frame to numpy float32 mono
        arr = frame.to_ndarray()
        if arr.ndim > 1:
            arr = np.mean(arr, axis=0)
        frames.append(arr)

    container.close()
    if not frames:
        raise ValueError(f"No audio frames decoded from {file_path}")

    waveform = np.concatenate(frames).astype(np.float32)
    # Normalize integer PCM types if needed
    if waveform.max() > 1.0 or waveform.min() < -1.0:
        waveform = waveform / 32768.0

    return waveform, sample_rate


def load_audio(file_path):
    """
    Load an audio file and return (waveform, sample_rate).

    waveform: 1-D float32 NumPy array, values roughly in [-1.0, 1.0].
              If the source file has multiple channels, they are averaged
              down to mono.
    sample_rate: int, samples per second of the ORIGINAL file.
    """
    try:
        waveform, sample_rate = sf.read(file_path, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        return waveform, sample_rate
    except Exception:
        # Fallback to PyAV (handles CAF, Apple IMA ADPCM, MP3, 3GP, AAC, etc.)
        return load_audio_av(file_path)
