import numpy as np
import sounddevice as sd
import soundfile as sf


def record_audio(sample_rate: int = 44100, channels: int = 1) -> np.ndarray:
    """Record audio from microphone between key presses."""

    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        recording.append(indata.copy())

    # Wait for 'r' to start
    input("To start recording press enter.")

    # Start streaming
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        input("Press 'enter' to stop recording.")

    print("Recording complete!")

    # Convert list of arrays to single array
    audio_data = np.concatenate(recording, axis=0)

    return audio_data


def play_audio(audio_data: np.ndarray, sample_rate: int = 44100):
    """Play audio with padding for better playback."""
    padding = np.zeros(int(sample_rate * 0.15))  # 150ms of silence
    audio_np = np.concatenate([padding, audio_data])
    sd.play(audio_np, samplerate=sample_rate, blocking=False)


if __name__ == "__main__":
    audio_data = record_audio()
    print("Recorded audio data shape:", audio_data.shape)
    play_audio(audio_data)
    sf.write("output.wav", audio_data, 44100)
