from abc import ABC, abstractmethod

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from kokoro import KPipeline
from torch import FloatTensor


class TextToSpeechModel(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> FloatTensor:
        """Convert text to speech and return audio data as a FloatTensor."""
        pass

    def play(self, audio: FloatTensor):
        """Play the synthesized audio."""
        pass

    def save(self, audio: FloatTensor, filename: str, sample_rate: int = 24000):
        """Save the synthesized audio to a file."""
        audio_np = audio.numpy()
        sf.write(filename, audio_np, samplerate=sample_rate)


class KokoroTTSModel(TextToSpeechModel):
    def __init__(self, lang_code: str = "a"):
        self.pipeline = KPipeline(lang_code=lang_code)

    def synthesize(self, text: str, voice: str = "af_heart") -> FloatTensor:
        generator = self.pipeline(text, voice=voice)
        # Get the first generated audio
        for i, (_, _, audio) in enumerate(generator):
            if isinstance(audio, FloatTensor):
                return audio
        raise ValueError("No audio generated from the given text")

    def play(self, audio: FloatTensor):
        audio_np = audio.numpy()
        samplerate = 24000
        padding = np.zeros(int(samplerate * 0.15))  # 150ms of silence
        audio_np = np.concatenate([padding, audio_np])
        sd.play(audio_np, samplerate=samplerate, blocking=False)


if __name__ == "__main__":
    tts_model = KokoroTTSModel()
    text = "Hello, this is a test of the Kokoro TTS model."
    output_file = "test_output.wav"
    audio_data = tts_model.synthesize(text)
    print("Synthesized audio data type:", type(audio_data))
    tts_model.play(audio_data)
    tts_model.save(audio_data, output_file)

    while True:
        user_input = input("Enter text to synthesize (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        audio_data = tts_model.synthesize(user_input)
        tts_model.play(audio_data)
        tts_model.save(audio_data, output_file)
