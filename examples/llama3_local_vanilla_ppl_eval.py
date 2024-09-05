import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..', 'models/vanilla/llama')
sys.path.append(module_path)

from modeling_llama import LlamaForCausalLM

from transformers import AutoTokenizer

import ppl_evaluator

remote_model_id = 'meta-llama/Meta-Llama-3-8B'
local_model_id = './pretrained_weights/vanilla_llama3_8b_fp16'

model = LlamaForCausalLM.from_pretrained(local_model_id, torch_dtype=torch.float16, device_map='auto', attn_implementation='eager')

from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(remote_model_id)
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
evaluator = ppl_evaluator.Evaluator(dataset, tokenizer, 'cuda')

ppl = evaluator.evaluate(model)
print(f"Local Vanilla Llama3-8B (fp16) perplexity: {ppl}")

