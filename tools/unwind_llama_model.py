import torch

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..')
sys.path.append(module_path)



remote_model_id = 'meta-llama/Meta-Llama-3-8B'
local_model_id = 'jpyo0803/sq-llama3-8b'
export_model_id = './pretrained_weights/Meta-Llama-3-8B-unwound-smoothquant.pt'

from models.quantized.llama.smoothquant_modeling_llama import SqLlamaForCausalLM

model = SqLlamaForCausalLM.from_pretrained(local_model_id, torch_dtype=torch.float16, device_map='cpu', attn_implementation='eager')

for idx in range(model.config.num_hidden_layers):
    decoder_layer = model.model.layers[idx]
    self_attn = model.model.layers[idx].self_attn

    decoder_layer.q_proj = self_attn.q_proj
    decoder_layer.k_proj = self_attn.k_proj
    decoder_layer.v_proj = self_attn.v_proj
    decoder_layer.o_proj = self_attn.o_proj

    decoder_layer.register_buffer("q_output_scale", self_attn.q_output_scale)
    decoder_layer.register_buffer("k_output_scale", self_attn.k_output_scale)
    decoder_layer.register_buffer("v_output_scale", self_attn.v_output_scale)

    del model.model.layers[idx].self_attn

    mlp = model.model.layers[idx].mlp

    decoder_layer.gate_proj = mlp.gate_proj
    decoder_layer.up_proj = mlp.up_proj
    decoder_layer.down_proj = mlp.down_proj

    del model.model.layers[idx].mlp


print("Unwound Model: ", model)

model.save_pretrained(export_model_id)

# Evaluate resulting model

from transformers import AutoTokenizer
from datasets import load_dataset

from models.quantized.llama.unwound_smoothquant_modeling_llama import UnwoundSqLlamaForCausalLM

unwound_model = UnwoundSqLlamaForCausalLM.from_pretrained(export_model_id, torch_dtype=torch.float16, device_map='cpu', attn_implementation='eager')
tokenizer = AutoTokenizer.from_pretrained(remote_model_id)
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
  
import examples.ppl_evaluator 

evaluator = examples.ppl_evaluator.Evaluator(dataset, tokenizer, unwound_model.device, n_samples=10)
ppl = evaluator.evaluate(unwound_model)

print(f"Local Unwound Smoothquant Llama3-8B (int8) perplexity: {ppl}")

# Expected PPL: 6.251369953155518
# Peak CPU memory usage: 15629.539 MB
# Peak GPU memory usage: 8588.000 MB