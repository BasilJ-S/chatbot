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
from chatbot.utils.chatbot_logger import logger

if __name__ == "__main__":
    stt_model = WhisperSTTModel()
    tts_model = KokoroTTSModel()
    llm = HuggingFaceDirectLM(FACEBOOK_MOBILELLM_CONFIG)
    chat_session = ChatSession(
        llm=llm,
        system_prompt="You are a helpful voice assistant. Keep your responses concise, very short, and unformatted, suitable for spoken delivery. Add very infrequent filler words like um or uh to make speech sound natural. Do not include any emojis.",
    )

    SAMPLE_RATE = stt_model.sample_rate

    logger.info("Please speak your prompt after pressing enter.")
    while True:
        audio = record_audio(sample_rate=SAMPLE_RATE).squeeze()
        transcription = stt_model.transcribe(audio)
        logger.info(f"Transcription: {transcription}")
        print(f"You said: {transcription}")
        if transcription.lower().strip(" ,.") in ["exit", "quit", "stop"]:
            break
        response = chat_session.chat(transcription)
        logger.info(f"LLM Response: {response}")
        print(f"Assistant: {response}")
        synthesized_audio = tts_model.synthesize(response)
        play_audio(synthesized_audio.numpy(), sample_rate=tts_model.sample_rate)
    logger.info("Exiting conversation loop.")

    logger.info("Generating conversation summary...")
    print("Generating conversation summary and saving history...")
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
    logger.info("Formatted Conversation History:")
    logger.info(formatted_history)

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

    logger.info(f"Conversation Summary: {summary}")

    two_words = "_".join(two_word_summary.lower().replace(",", " ").split())
    logger.info(f"Two-Word Summary: {two_words}")

    # Save final conversation history to a text file
    with open(
        f"conversation_history/history_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{two_words}.txt",
        "w",
    ) as f:
        f.write("Conversation Summary:\n")
        f.write(summary + "\n\n")
        f.write("Full Conversation History:\n\n")
        f.write(formatted_history)
    logger.info(
        f"Conversation history saved to conversation_history_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    print("Conversation history and summary saved.\nSUMMARY:\n", summary)
    synthesized_summary = tts_model.synthesize(summary)
    play_audio(
        synthesized_summary.numpy(), sample_rate=tts_model.sample_rate, wait=True
    )
