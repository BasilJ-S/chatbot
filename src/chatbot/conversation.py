import datetime as dt

from chatbot.models.llm import Quen3_4B
from chatbot.models.stt import WhisperSTTModel
from chatbot.models.tts import KokoroTTSModel
from chatbot.utils.audio import play_audio, record_audio

if __name__ == "__main__":
    stt_model = WhisperSTTModel()
    tts_model = KokoroTTSModel()
    llm = Quen3_4B(
        system_prompt="You are a helpful assistant engaged in a voice conversation with a user. Keep your responses concise, very short, and unformatted, suitable for spoken delivery. Add very infrequent filler words like um or uh to make speech sound natural. Do not include any emojis."
    )

    SAMPLE_RATE = stt_model.sample_rate

    print("Please speak your prompt after pressing enter.")
    while True:
        audio = record_audio(sample_rate=SAMPLE_RATE).squeeze()
        transcription = stt_model.transcribe(audio)
        print("Transcription:", transcription)
        if transcription.lower().strip(" ,.") in ["exit", "quit", "stop"]:
            break
        response = llm.chat(transcription)
        print("LLM Response:", response)
        synthesized_audio = tts_model.synthesize(response)
        play_audio(synthesized_audio.numpy(), sample_rate=tts_model.sample_rate)
    print("Exiting conversation loop.")

    summary = llm.chat(
        "Provide a brief formatted summary of our conversation, being sure to include key points discussed. Keep it concise, and do not include filler words."
    )
    print("Conversation Summary:")
    print(summary)

    two_words = "_".join(
        llm.chat(
            "Provide a two-word or three-word summary that captures the main discussion points of our conversation. Only respond with the two or three words. Do not include more than three words."
        )
        .lower()
        .split(" ")
    )
    print("Two-Word Summary:", two_words)

    # Save final conversation history to a text file
    with open(
        f"conversation_history/history_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{two_words}.txt",
        "w",
    ) as f:
        f.write("Conversation Summary:\n")
        f.write(summary + "\n\n")
        f.write("Full Conversation History:\n\n")
        for message in llm.history:
            f.write(f"{message['role']}: {message['content']}\n")
    print(
        f"Conversation history saved to conversation_history_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    synthesized_summary = tts_model.synthesize(summary)
    play_audio(
        synthesized_summary.numpy(), sample_rate=tts_model.sample_rate, wait=True
    )
