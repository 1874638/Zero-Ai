import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROMPT_TMPL_WITH_INPUT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
PROMPT_TMPL_NO_INPUT = """### Instruction:
{instruction}

### Response:
"""

def build_prompt(instruction: str, input_text: str = "") -> str:
    if input_text and len(input_text.strip()) > 0:
        return PROMPT_TMPL_WITH_INPUT.format(instruction=instruction.strip(), input=input_text.strip())
    return PROMPT_TMPL_NO_INPUT.format(instruction=instruction.strip())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter_dir", type=str, required=True, help="path to LoRA adapter (output_dir)")
    parser.add_argument("--instruction", type=str, default="요약해줘: 딥러닝은...")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    prompt = build_prompt(args.instruction, args.input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the prompt part to show only the response
    if "### Response:" in text:
        print(text.split("### Response:")[-1].strip())
    else:
        print(text)

if __name__ == "__main__":
    main()