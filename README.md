# Chatbot — long-form conversational assistant

Small, opinionated starter project for a chatbot focused on long-form, multi-turn conversations with explicit, configurable memory management.

## Features

- **Speech-to-speech interaction**: Speak naturally and get spoken responses
- **Multi-turn conversations**: Maintain context across extended dialogues
- **Automatic conversation summaries**: Generated at session end and saved to disk
- **Modular architecture**: Swap STT, TTS, or LLM models easily

## Purpose

- Support extended, coherent conversations (not a series of unrelated short chats).
- Provide a clear memory model that can be iterated on and swapped out.
- Be simple enough to extend and experiment with different memory strategies.

## Key goals

- Maintain useful long-term and short-term context.
- Keep response relevance and user intent continuity across sessions.
- Make memory behavior observable and configurable.
- Minimal, modular code so different memory strategies can be tested.

## Setup

Depends on espeak-ng. Install with

```
brew install espeak-ng
```

This is a dependency of kokoro, so may not stay indefinitely.

Also requires pip to be installed for kokoro running, which requires

```
uv venv --seed
```
