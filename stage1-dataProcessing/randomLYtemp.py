
from transformers import AutoTokenizer
import transformers
import torch
from typing import List, Tuple, Optional, Dict, Union, Literal
from langchain.llms.base import LLM

# import tensor_parallel as tp
from easyllm.prompt_utils.llama2 import build_llama2_prompt
from easyllm.schema.base import ChatMessage
# from augmentor.texts import Llama2
class Llama2(LLM):
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 2048
    tokenizer: Any
    model: Any

    def __init__(self, model_name_or_path, temperature, max_tokens, bit4=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if bit4 == False:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=False,
            )
            self.model.eval()
        else:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self.model.eval()
            # from auto_gptq import AutoGPTQForCausalLM
            # self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,low_cpu_mem_usage=True, device="cuda:0", use_triton=False,inject_fused_attention=False,inject_fused_mlp=False)

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     self.model = torch.compile(self.model)

    @property
    def _llm_type(self) -> str:
        return "Llama2"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # print('prompt:',prompt)
        # prompt = build_llama2_prompt(prompt)

        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to("cuda")
        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_tokens,
            "do_sample": True,
            "top_k": 50,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": 1.2,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        generate_ids = self.model.generate(**generate_input)
        generate_ids = [item[len(input_ids[0]) : -1] for item in generate_ids]
        result_message = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return result_message
# model = "meta-llama/Llama-2-7b-chat-hf"
# model = "NousResearch/Llama-2-13b-chat-hf"
# model = "NousResearch/Llama-2-7b-chat-hf"
# model = "NousResearch/Llama-2-70b-chat-hf"
model = "meta-llama/Llama-2-70b-chat-hf"

relation = "per:country_of_birth"
description = "The country in which the assigned person was born."

prompt = f"Your task is to produce 50 sentences with labels that illustrates the relation {relation} for the purpose of relation extraction. The definition of {relation} is: {description} Please wrap the head entity with [h] and [/h] tags, the tail entity with [t] and [/t] tags, and should not explicitly mention the relation label '{relation}'. Write one sample per line in json format, e.g., {{'index': 0, 'relation': '{relation}', 'sentence': '...'}}. The head and tail entities may appear in any place in a sentence. No other output."
input_prompt = build_llama2_prompt(prompt)
print(input_prompt)


# tokenizer = transformers.AutoTokenizer.from_pretrained(model)
# model = transformers.AutoModelForCausalLM.from_pretrained(
#     model, device_map="auto", torch_dtype=torch.float16, temperature=1.2, max_length=200
# )  # use opt-125m for testing

# system = "You are a experienced data annotator in relation extraction domain. You are writing sentences for preparing a dataset for relation extraction."
# user = "Write a sentence about relation per:country_of_birth, you need to wrap the head entity by [h], [/h], and the tail entity by [t], [/t]. The sentence should start with 'In'. Please output the result directly."
# inputs = [
#     {"role": "system", "content": system},
#     {"role": "user", "content": user},
# ]

# messages = [ChatMessage(**message) for message in inputs]
# inputs = build_llama2_prompt(messages)
# inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to("cuda")

# outputs = model.generate(inputs)
# print(tokenizer.decode(outputs[0]))  # A cat sat on my lap for a few minutes ...

llm = Llama2(model_name_or_path=model, temperature=1.2, max_tokens=1024)
# prompt_params = {"relation": "per:country_of_birth", "description": "The country in which the assigned person was born."}

# input_prompt = prompt_template.format(**prompt_params)
# print(text_config)

output = llm.predict(input_prompt)
print(output)