import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from transformers.models.llama.modeling_llama import (
  LlamaForCausalLM,
)

from transformers import AutoTokenizer

import ppl_evaluator

model_id = 'meta-llama/Meta-Llama-3-8B'

model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto', attn_implementation='eager')

from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
evaluator = ppl_evaluator.Evaluator(dataset, tokenizer, 'cuda')

ppl = evaluator.evaluate(model)
print(f"Remote Vanilla Llama3-8B (fp16) perplexity: {ppl}")
