import datetime as dt

from chatbot.models.llm import (
    DOLPHIN_LLAMA_CONFIG,
    FACEBOOK_MOBILELLM_CONFIG,
    QUEN3_4B_CONFIG,
    ChatSession,
    HuggingFaceDirectLM,
)
from chatbot.models.stt import WhisperSTTModel
from chatbot.models.tts import KokoroTTSModel
from chatbot.utils.audio import play_audio, record_audio

if __name__ == "__main__":
    stt_model = WhisperSTTModel()
    tts_model = KokoroTTSModel()
    llm = HuggingFaceDirectLM(QUEN3_4B_CONFIG)
    chat_session = ChatSession(
        llm=llm,
        system_prompt="You are a helpful voice assistant. Keep your responses concise, very short, and unformatted, suitable for spoken delivery. Add very infrequent filler words like um or uh to make speech sound natural. Do not include any emojis.",
    )

    SAMPLE_RATE = stt_model.sample_rate

    print("Please speak your prompt after pressing enter.")
    while True:
        audio = record_audio(sample_rate=SAMPLE_RATE).squeeze()
        transcription = stt_model.transcribe(audio)
        print("Transcription:", transcription)
        if transcription.lower().strip(" ,.") in ["exit", "quit", "stop"]:
            break
        response = chat_session.chat(transcription)
        print("LLM Response:", response)
        synthesized_audio = tts_model.synthesize(response)
        play_audio(synthesized_audio.numpy(), sample_rate=tts_model.sample_rate)
    print("Exiting conversation loop.")

    print("Generating conversation summary...")
    history_without_system = [
        message
        for message in chat_session.memory.history
        if message["role"] != "system"
    ].copy()
    formatted_history = "\n".join(
        [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in history_without_system
        ]
    )
    print("Formatted Conversation History:")
    print(formatted_history)

    summary_chat = ChatSession(
        llm=llm,
        system_prompt="You are an expert summarizer. You are concise, and provide formatted summaries showing the key points discussed in each section of the conversation.",
    )
    summary = summary_chat.chat(
        f"Please summarize the following conversation:\n{formatted_history}"
    )

    two_word_chat = ChatSession(
        llm=llm,
        system_prompt="You are an expert summarizer. You provide two or three word comma separated summaries of conversations. For example, if the conversation is about planning a trip to Paris, you might respond with 'paris,trip'.",
    )
    two_word_summary = two_word_chat.chat(
        f"Please summarize the following conversation in two or three words:\n{formatted_history}"
    )

    print("Conversation Summary:")
    print(summary)
    print("Two-Word Summary:")
    print(two_word_summary)

    two_words = "_".join(two_word_summary.lower().replace(",", " ").split())
    print("Two-Word Summary:", two_words)

    # Save final conversation history to a text file
    with open(
        f"conversation_history/history_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{two_words}.txt",
        "w",
    ) as f:
        f.write("Conversation Summary:\n")
        f.write(summary + "\n\n")
        f.write("Full Conversation History:\n\n")
        f.write(formatted_history)
    print(
        f"Conversation history saved to conversation_history_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    synthesized_summary = tts_model.synthesize(summary)
    play_audio(
        synthesized_summary.numpy(), sample_rate=tts_model.sample_rate, wait=True
    )
