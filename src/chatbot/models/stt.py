from abc import ABC, abstractmethod
from platform import processor

import numpy as np
import torch
from kokoro import model
from numpy.ma import masked
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from chatbot.utils.audio import record_audio


class SpeechToTextModel(ABC):

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe the given audio to text."""
        pass


class WhisperSTTModel(SpeechToTextModel):
    def __init__(self, model_name: str = "openai/whisper-small") -> None:
        # load model and processor
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.sample_rate = 16000

    def transcribe(self, audio: np.ndarray) -> str:
        # preprocess the audio
        input_features = self.processor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features

        input_mask = torch.ones(input_features.shape, dtype=torch.long)
        # generate token ids
        predicted_ids = self.model.generate(
            input_features,
            language="english",
            task="transcribe",
            attention_mask=input_mask,
        )

        # transcribe
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        return transcription[0]


if __name__ == "__main__":
    stt_model = WhisperSTTModel()
    SAMPLE_RATE = stt_model.sample_rate

    audio = record_audio(sample_rate=SAMPLE_RATE).squeeze()
    transcription = stt_model.transcribe(audio)
    print("Final Transcription:", transcription)
