from abc import ABC, abstractmethod

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from kokoro import KPipeline
from torch import FloatTensor

from chatbot.utils.audio import play_audio


class TextToSpeechModel(ABC):

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the sample rate of the audio."""
        pass

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
        self._sample_rate = 24000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str, voice: str = "af_heart") -> FloatTensor:
        generator = self.pipeline(text, voice=voice)
        # Get the first generated audio
        for i, (_, _, audio) in enumerate(generator):
            if isinstance(audio, FloatTensor):
                return audio
        raise ValueError("No audio generated from the given text")


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
        play_audio(audio_data.numpy(), sample_rate=tts_model.sample_rate)
        sf.write(output_file, audio_data.numpy(), samplerate=tts_model.sample_rate)
