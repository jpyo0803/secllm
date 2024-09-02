import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from transformers.models.llama.modeling_llama import (
  LlamaForCausalLM,
)

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..', 'models/vanilla/llama')
sys.path.append(module_path)

from modeling_llama import LlamaForCausalLM

from transformers import AutoTokenizer

import ppl_evaluator

model_id = 'meta-llama/Meta-Llama-3-8B'

model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')

from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
evaluator = ppl_evaluator.Evaluator(dataset, tokenizer, 'cuda')

ppl = evaluator.evaluate(model)
print(f"Local Vanilla Llama3-8B (fp16) perplexity: {ppl}")

