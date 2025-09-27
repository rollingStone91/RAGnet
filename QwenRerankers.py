import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Dict, Optional, List
import json
import logging
import torch
from transformers import AutoTokenizer, is_torch_npu_available
# from vllm import LLM, SamplingParams
# from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import math
# from vllm.inputs.data import TokensPrompt

# transformer版本
class QwenReranker:
    def __init__(self, model_name="./models/Qwen3-Reranker-0.6B", max_length=8192, device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                           torch_dtype=torch.float16,
                                                           attn_implementation="flash_attention_2",
                                                           trust_remote_code=True).to(device).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = max_length
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

# class QwenReranker:
#     def __init__(self, model_name="./models/Qwen3-Reranker-0.6B", max_length=8192,
#                   suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
#                   task = 'Given a web search query, retrieve relevant passages that answer the query'):
#         number_of_gpu = torch.cuda.device_count()
#         print(f"number_of_gpu: {number_of_gpu}")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = LLM(model=model_name, tensor_parallel_size=number_of_gpu, max_model_len=10000, enable_prefix_caching=True, gpu_memory_utilization=0.4)
#         self.max_length = max_length

#         self.tokenizer.padding_side = "left"
#         self.tokenizer.pad_token = self.tokenizer.eos_token

#         self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

#         self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
#         self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
#         self.sampling_params = SamplingParams(temperature=0, 
#             max_tokens=1,
#             logprobs=20, 
#             allowed_token_ids=[self.true_token, self.false_token],
#         )

#         self.task = task

#     def format_instruction(self, query, doc):
#         text = [
#             {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
#             {"role": "user", "content": f"<Instruct>: {self.task}\n\n<Query>: {query}\n\n<Document>: {doc}"}
#         ]
#         return text

#     def process_inputs(self, pairs):
#         max_length = self.max_length - len(self.suffix_tokens)
#         messages = [self.format_instruction(query, doc) for query, doc in pairs]
#         messages =  self.tokenizer.apply_chat_template(
#             messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
#         )
#         messages = [ele[:max_length] + self.suffix_tokens for ele in messages]
#         messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
#         return messages

#     def compute_logits(self, messages):
#         outputs = self.model.generate(messages, self.sampling_params, use_tqdm=False)
#         scores = []
#         for i in range(len(outputs)):
#             final_logits = outputs[i].outputs[0].logprobs[-1]
#             token_count = len(outputs[i].outputs[0].token_ids)
#             if self.true_token not in final_logits:
#                 true_logit = -10
#             else:
#                 true_logit = final_logits[self.true_token].logprob
#             if self.false_token not in final_logits:
#                 false_logit = -10
#             else:
#                 false_logit = final_logits[self.false_token].logprob
#             true_score = math.exp(true_logit)
#             false_score = math.exp(false_logit)
#             score = true_score / (true_score + false_score)
#             scores.append(score)
#         return scores


# if __name__ == "__main__":
#     task = 'Given a web search query, retrieve relevant passages that answer the query'
#     queries = ["What is the capital of China?",
#         "Explain gravity",
#     ]

#     documents = [
#         "The capital of China is Beijing.",
#         "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
#     ]

#     reranker = QwenReranker()
#     pairs = [reranker.format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

#     # Tokenize the input texts
#     inputs = reranker.process_inputs(pairs)
#     scores = reranker.compute_logits(inputs)

#     print("scores: ", scores)
