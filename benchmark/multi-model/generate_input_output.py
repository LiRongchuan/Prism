# Example

# python -m generate_input_output

# python -m generate_input_output \
# --model_list meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.2-1B  \
# --N 100 \
# --mu_i 10.0 \
# --sigma_i 2.0 \
# --batch_size 8 \
# --output_file model_outputs.json \
# --profile_file profile.json \
# --max_new_tokens 50

import argparse
import json
import time
from dataclasses import dataclass
from statistics import mean, stdev

import numpy as np
import torch
from transformers import AutoTokenizer

import sglang as sgl


@dataclass
class ExperimentConfig:
    model_list: list
    N: int = 100
    mu_i: float = 10.0
    sigma_i: float = 2.0
    batch_size: int = 8
    sampling_params: dict = None
    output_file: str = "model_outputs.json"
    profile_file: str = "profile.json"
    max_new_tokens: int = 50


def generate_inputs(N, mu_i, sigma_i):
    inputs = []
    for _ in range(N):
        length = int(np.random.normal(mu_i, sigma_i))
        length = max(1, length)
        input_text = " ".join(["hello"] * length)
        inputs.append(input_text)
    return inputs


def process_model_outputs(
    model_path, inputs, batch_size, sampling_params, max_new_tokens
):
    llm = sgl.Engine(model_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    results = []

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        outputs = llm.generate(
            batch, {**sampling_params, "max_new_tokens": max_new_tokens}
        )
        for input_text, output in zip(batch, outputs):
            output_text = output["text"]
            input_length = len(input_text.split())
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            results.append(
                {
                    "model": model_path,
                    "input": input_text,
                    "input_length": input_length,
                    "output": output_text,
                    "output_tokens": output_tokens,
                }
            )

    return results


def calculate_statistics(results):
    profiles = {}
    for model in set([res["model"] for res in results]):
        model_results = [res for res in results if res["model"] == model]
        input_lengths = [res["input_length"] for res in model_results]
        output_tokens = [res["output_tokens"] for res in model_results]

        mu_i = mean(input_lengths)
        sigma_i = stdev(input_lengths) if len(input_lengths) > 1 else 0.0
        mu_o = mean(output_tokens)
        sigma_o = stdev(output_tokens) if len(output_tokens) > 1 else 0.0

        profiles[model] = {
            "mu_i": mu_i,
            "sigma_i": sigma_i,
            "mu_o": mu_o,
            "sigma_o": sigma_o,
        }
    return profiles


def main(config: ExperimentConfig):
    inputs = generate_inputs(config.N, config.mu_i, config.sigma_i)
    all_results = []

    for model_path in config.model_list:
        print(f"Processing model: {model_path}")
        st = time.time()
        model_results = process_model_outputs(
            model_path,
            inputs,
            config.batch_size,
            config.sampling_params,
            config.max_new_tokens,
        )
        print(f"E2E Latency for {model_path}: {time.time() - st} seconds")
        torch.cuda.empty_cache()
        all_results.extend(model_results)

    with open(config.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {config.output_file}")

    profiles = calculate_statistics(all_results)
    with open(config.profile_file, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=4)
    print(f"Profile saved to {config.profile_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run input-output analysis for language models."
    )
    parser.add_argument("--model_list", nargs="+", default=["meta-llama/Llama-3.2-1B"])
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--mu_i", type=float, default=10.0)
    parser.add_argument("--sigma_i", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_file", type=str, default="model_outputs.json")
    parser.add_argument("--profile_file", type=str, default="profile.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": args.max_new_tokens,
    }
    config = ExperimentConfig(
        model_list=args.model_list,
        N=args.N,
        mu_i=args.mu_i,
        sigma_i=args.sigma_i,
        batch_size=args.batch_size,
        sampling_params=sampling_params,
        output_file=args.output_file,
        profile_file=args.profile_file,
        max_new_tokens=args.max_new_tokens,
    )
    main(config)
