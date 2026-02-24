#!/usr/bin/env python3
# inference.py
# Inference / text generation script for DeepSeek-Replica

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from model import DeepSeekForCausalLM
from tokenizer.tokenizer import DeepSeekTokenizer


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_dir: str,
    device: Optional[str] = None,
) -> tuple:
    """Load a saved checkpoint and tokenizer."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer from {tokenizer_dir} ...")
    tokenizer = DeepSeekTokenizer.load(tokenizer_dir)

    print(f"Loading model from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    config.use_cache = True

    model = DeepSeekForCausalLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    print(f"Model loaded (step {ckpt.get('step', '?')})")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: DeepSeekForCausalLM,
    tokenizer: DeepSeekTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    device: str = "cpu",
    stream: bool = True,
) -> str:
    """Generate text from a prompt string."""
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"\n--- Prompt ---\n{prompt}\n--- Generation ---")

    if stream:
        # Streaming token-by-token output
        generated_ids = list(input_ids)
        past_kv = None
        for _ in range(max_new_tokens):
            out = model(
                input_tensor if past_kv is None else input_tensor[:, -1:],
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out["past_key_values"]
            logits = out["logits"][:, -1, :] / max(temperature, 1e-5)

            # Top-K
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            # Top-P
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = sorted_logits.softmax(-1).cumsum(-1)
                remove = cum_probs - sorted_logits.softmax(-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = logits.softmax(-1)
            next_id = torch.multinomial(probs, 1).item()
            generated_ids.append(next_id)
            input_tensor = torch.tensor([[next_id]], device=device)

            # Stream decoded token
            token_str = tokenizer.decode([next_id], skip_special_tokens=False)
            print(token_str, end="", flush=True)

            if next_id == tokenizer.eos_token_id:
                break
        print()
        return tokenizer.decode(generated_ids[len(input_ids):], skip_special_tokens=True)
    else:
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = output_ids[0, len(input_ids):].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(text)
        return text


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def interactive_loop(
    model: DeepSeekForCausalLM,
    tokenizer: DeepSeekTokenizer,
    device: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
):
    """Run an interactive chat/generation loop in the terminal."""
    print("\nDeepSeek-Replica Inference REPL")
    print("Type your prompt and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if prompt.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not prompt:
            continue
        generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            device=device,
            stream=True,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepSeek-Replica inference")
    parser.add_argument("--checkpoint",    required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--tokenizer_dir", required=True, help="Path to saved tokenizer directory")
    parser.add_argument("--prompt",        default=None,  help="Prompt string (omit for interactive REPL)")
    parser.add_argument("--max_new_tokens", type=int,  default=200)
    parser.add_argument("--temperature",   type=float, default=0.7)
    parser.add_argument("--top_p",         type=float, default=0.9)
    parser.add_argument("--top_k",         type=int,   default=50)
    parser.add_argument("--device",        default=None)
    parser.add_argument("--no_stream",     action="store_true")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(
        args.checkpoint, args.tokenizer_dir, args.device
    )

    if args.prompt:
        generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
            stream=not args.no_stream,
        )
    else:
        interactive_loop(
            model, tokenizer, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
