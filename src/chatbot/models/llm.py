from abc import ABC, abstractmethod

from sympy.printing.pytorch import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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


class Quen3_4B(LanguageModel):
    def __init__(
        self,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
    ):
        model_name = "Qwen/Qwen3-4B-Instruct-2507"

        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.min_p = min_p

    def generate(self, messages: list[dict[str, str]]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print("Model input text:", text)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content


if __name__ == "__main__":

    # prepare the model input
    prompt = "Give me a short introduction to large language model."
    llm = Quen3_4B()
    chat_session = ChatSession(llm=llm, system_prompt="You are a helpful assistant.")
    response = chat_session.chat(prompt)
    print("Response from Quen3-4B model:")
    print(response)
    while True:
        user_input = input("Enter your prompt (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = chat_session.chat(user_input)
        print("Response:")
        print(response)
    print("Exiting...")
    print("Final conversation history:")
    for message in chat_session.memory.history:
        print(f"{message['role']}: {message['content']}")
