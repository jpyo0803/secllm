import torch
import argparse
import os

from pathlib import Path

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..', 'models', 'quantized', 'llama')
sys.path.append(module_path)

from smoothquant_modeling_llama import SqLlamaForCausalLM

module_path = os.path.join(current_dir, '..', 'quantization')
sys.path.append(module_path)

from smooth import smooth_lm

from calibration import get_static_decoder_layer_scales

module_path = os.path.join(current_dir, '..', 'examples')
sys.path.append(module_path)

import ppl_evaluator

from datasets import load_dataset

def ExportInt8Model(push_to_hub=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='meta-llama/Meta-Llama-3-8B')
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--act-scales", type=str,
                        default='act_scales/llama3-8b.pt')
    parser.add_argument("--output-path", type=str, default='pretrained_weights')
    parser.add_argument('--dataset-path', type=str, default='datasets/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    # parser.add_argument('--export-FT', default=False, action="store_true")
    args = parser.parse_args()
    model = LlamaForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16, attn_implementation='eager')

    act_scales = torch.load(args.act_scales)
    smooth_lm(model, act_scales, 0.85) # Llama3 uses 0.85
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError
    
    decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(model,
                                                                       tokenizer,
                                                                       args.dataset_path,
                                                                       num_samples=args.num_samples,
                                                                       seq_len=args.seq_len)
    output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant.pt")
    # if args.export_FT:
    #     model.save_pretrained(output_path)
    #     print(f"Saved smoothed model at {output_path}")

    #     output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant-scales.pt")
    #     torch.save(raw_scales, output_path)
    #     print(f"Saved scaling factors at {output_path}")
    # else:
    int8_model = SqLlamaForCausalLM.from_float(model, decoder_layer_scales)

    print("int8 model: ")
    print(int8_model)

    int8_model.save_pretrained(output_path)
    print(f"Saved int8 model at {output_path}")

    if push_to_hub:
        int8_model.push_to_hub('jpyo0803/sq-llama3-8b')
        print(f"Pushed int8 model to the hub")

    return int8_model, output_path, tokenizer

if __name__ == '__main__':
    int8_model, output_path, tokenizer = ExportInt8Model(push_to_hub=False)

    # NOTE(jpyo0803): Seems like the 'auto' option in 'ExportInt8Model' arbitrarily assigns some module in cpu
    int8_model = int8_model.to('cuda:0')

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    evaluator = ppl_evaluator.Evaluator(dataset, tokenizer, 'cuda')

    ppl = evaluator.evaluate(int8_model)
    print(f"Smoothquantized Llama3-8B (int8) perplexity: {ppl}")