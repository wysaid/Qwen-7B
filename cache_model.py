# coding=utf-8
# Implements API for Qwen-7B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.

from argparse import ArgumentParser
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.generation import GenerationConfig
# from auto_gptq import AutoGPTQForCausalLM

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str,
                        default='Qwen/Qwen-7B-Chat',
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )
    if args.checkpoint_path == "Qwen/Qwen-7B-Chat-Int4":
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            args.checkpoint_path,
            device_map=device_map,
            trust_remote_code=True,
            resume_download=True,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            device_map=device_map,
            trust_remote_code=True,
            resume_download=True,
        ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    sys.exit(0)
