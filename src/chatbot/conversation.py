import datetime as dt

from chatbot.models.llm import (
    DOLPHIN_LLAMA_CONFIG,
    FACEBOOK_MOBILELLM_CONFIG,
    QUEN3_4B_CONFIG,
    SMOLLLM_CONFIG,
    ChatSession,
    HuggingFaceDirectLM,
)
from chatbot.models.stt import WhisperSTTModel
from chatbot.models.tts import KokoroTTSModel
from chatbot.utils.audio import play_audio, record_audio
from chatbot.utils.chatbot_logger import logger


def format_message_history(history: list[dict[str, str]]) -> str:
    """Format the message history for logging or display."""
    formatted_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history]
    )
    return formatted_history


if __name__ == "__main__":
    stt_model = WhisperSTTModel()
    tts_model = KokoroTTSModel()

    analysis_llm_config = QUEN3_4B_CONFIG
    llm_config = SMOLLLM_CONFIG
    llm = HuggingFaceDirectLM(llm_config)
    analysis_llm = HuggingFaceDirectLM(analysis_llm_config)
    analysis_chat = ChatSession(
        llm=analysis_llm,
        system_prompt="You analyze conversations and provide advice on how the assistant can improve its responses. You keep responses to one short sentence in length. You focus on the most recent user-assistant exchanges. If the assistant performed well, give an empty response. If the assistant forgot something, remind them of it.",
    )
    memory_chat = ChatSession(
        llm=analysis_llm,
        system_prompt="You are an expert at managing conversation memory for a voice assistant. You help keep the assistant's memory relevant and concise by summarizing or removing less important details. You keep responses to one short sentence in length.",
        system_prompt_prefix=analysis_llm_config.system_prompt_prefix,
    )
    chat_session = ChatSession(
        llm=llm,
        system_prompt="You are a helpful voice assistant. Keep your responses concise, very short, and unformatted, suitable for spoken delivery. Add very infrequent filler words like um or uh to make speech sound natural. Do not include any emojis.",
        system_prompt_prefix=llm_config.system_prompt_prefix,
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
        if not response.strip():
            logger.info("Empty response from LLM, skipping TTS playback.")
            print("No response generated.")
            continue

        synthesized_audio = tts_model.synthesize(response)
        play_audio(synthesized_audio.numpy(), sample_rate=tts_model.sample_rate)

        # Try to improve assistant response while the user listens
        chat_history = format_message_history(
            chat_session.memory.history[1:]
        )  # Exclude system prompt
        analysis_response = analysis_chat.chat(
            f"Analyze the following conversation. Give one piece of advice on how to improve the conversation.\n{chat_history}"
        )
        memory_response = memory_chat.chat(
            f"Analyze the following conversation. Write a concise memory for the assistant.\n{chat_history}"
        )
        logger.info(f"Memory Response: {memory_response}")
        memory_chat.memory.clear()  # Clear memory to focus on recent context
        analysis_chat.memory.clear()  # Clear memory to focus on recent context
        advice = analysis_response
        chat_session.memory.update_system_prompt_in_history(
            f"Original Prompt:{chat_session.memory.system_prompt}. Feedback: {advice}. Memory: {memory_response}"
        )

        logger.info(f"Analysis Response: {analysis_response}")
        logger.info(
            f"System prompt updated with analysis advice. New system prompt: {chat_session.memory.history[0]}"
        )
        print(
            f"System prompt updated with analysis advice. New system prompt: {chat_session.memory.history[0]}"
        )
        print(f"Analysis: {analysis_response}")

    logger.info("Exiting conversation loop.")

    logger.info("Generating conversation summary...")
    print("Generating conversation summary and saving history...")
    history_without_system = [
        message
        for message in chat_session.memory.history
        if message["role"] != "system"
    ].copy()
    formatted_history = format_message_history(history_without_system)
    logger.info("Formatted Conversation History:")
    logger.info(formatted_history)

    summary_chat = ChatSession(
        llm=analysis_llm,
        system_prompt="You are an expert summarizer. You are concise, and provide formatted summaries showing the key points discussed in each section of the conversation.",
        system_prompt_prefix=analysis_llm_config.system_prompt_prefix,
    )
    summary = summary_chat.chat(
        f"Please summarize the following conversation:\n{formatted_history}"
    )

    two_word_chat = ChatSession(
        llm=analysis_llm,
        system_prompt="You are an expert summarizer. You provide two or three word comma separated summaries of conversations. For example, if the conversation is about planning a trip to Paris, you might respond with 'paris,trip'.",
        system_prompt_prefix=analysis_llm_config.system_prompt_prefix,
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
