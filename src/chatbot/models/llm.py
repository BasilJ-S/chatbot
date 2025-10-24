from abc import ABC, abstractmethod
from dataclasses import dataclass
from re import I

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from chatbot.utils.chatbot_logger import logger


class LanguageModel(ABC):

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response based on the given prompt and chat history."""
        pass


class ConversationMemory:
    def __init__(self, system_prompt: str = ""):
        self._history: list[dict[str, str]] = []
        self.system_prompt = system_prompt

        self._history.append({"role": "system", "content": system_prompt})

    @property
    def history(self) -> list[dict[str, str]]:
        return self._history.copy()

    def add_message(self, role: str, content: str):
        self._history.append({"role": role, "content": content})

    def clear(self):
        self._history = []
        if self.system_prompt:
            self._history.append({"role": "system", "content": self.system_prompt})

    def update_system_prompt_in_history(self, new_prompt: str):
        if self._history and self._history[0]["role"] == "system":
            self._history[0]["content"] = new_prompt
        else:
            self._history.insert(0, {"role": "system", "content": new_prompt})


class ChatSession:
    def __init__(self, llm: LanguageModel, system_prompt: str = ""):
        self.llm = llm
        self.memory = ConversationMemory(system_prompt=system_prompt)

    def chat(self, prompt: str) -> str:
        self.memory.add_message("user", prompt)
        messages = self.memory.history
        content = self.llm.generate(messages)
        self.memory.add_message("assistant", content)
        return content


@dataclass
class LLMConfig:
    model_name: str
    description: str = ""
    max_new_tokens: int | None = 200
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None


QUEN3_4B_CONFIG = LLMConfig(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    description="Qwen3-4B model fine-tuned for instruction following. A bit slow but good quality.",
    max_new_tokens=200,
    temperature=0.7,  # Parameters recommended in best practices for instruction following model.
    top_p=0.8,
    top_k=20,
    min_p=0.0,
)

FACEBOOK_MOBILELLM_CONFIG = LLMConfig(
    model_name="facebook/MobileLLM-Pro",
    description="Facebook MobileLLM Pro model optimized for mobile and edge devices. Fast, good for testing, low quality.",
)

DOLPHIN_LLAMA_CONFIG = LLMConfig(
    model_name="dphn/Dolphin3.0-Llama3.2-3B",
    description="Dolphin 3.0 LLaMA 3.2 3B model. Faster, lower quality, with safety barriers intentionally removed. Use with caution.",
)


class HuggingFaceDirectLM(LanguageModel):

    def __init__(self, config: LLMConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.config = config

    def generate(self, messages: list[dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            min_p=self.config.min_p,
        )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )


if __name__ == "__main__":

    # prepare the model input
    prompt = "Give me a short introduction to large language model."
    llm = HuggingFaceDirectLM(config=FACEBOOK_MOBILELLM_CONFIG)
    chat_session = ChatSession(llm=llm, system_prompt="You are a helpful assistant.")
    response = chat_session.chat(prompt)
    logger.info("Response from model:")
    logger.info(response)
    while True:
        user_input = input("Enter your prompt (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = chat_session.chat(user_input)
        logger.info("Response:")
        logger.info(response)
    logger.info("Exiting...")
    logger.info("Final conversation history:")
    for message in chat_session.memory.history:
        logger.info(f"{message['role']}: {message['content']}")
