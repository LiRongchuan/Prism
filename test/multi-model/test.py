import argparse

import requests


def sample_requests(model_paths, slo):
    prompts = [
        "What is the meaning of life?",
        "How do I make a cake",
        "What is the best way to learn a new language?",
        "Tell me a joke",
        "Where is the best place to go on vacation?",
        "How do I fix a leaky faucet?",
    ]

    sampling_params = {"temperature": 0.5, "top_p": 0.9, "top_k": 50}
    requests = []
    # distributed prompts to models
    num_models = len(model_paths)
    for i, prompt in enumerate(prompts):
        model_path = model_paths[i % num_models]
        requests.append(
            {
                "model": model_path,
                "text": prompt,
                "sampling_params": sampling_params,
                "slo": slo,
            }
        )
    return requests


def test(url, model_paths, slo):
    generation_requests = sample_requests(model_paths, slo)
    for gen_request in generation_requests:
        response = requests.post(f"{url}/generate", json=gen_request)
        if isinstance(response, str):
            ret = response
        else:
            ret = response.json()
        print(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--num-models", "-n", type=int, default=2)
    parser.add_argument(
        "--model-paths",
        "-m",
        type=str,
        nargs="+",
        help="The paths of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--slo", type=int, default=10)
    args = parser.parse_args()

    url = f"{args.host}:{args.port}"
    if args.model_paths is None:
        model_paths = [f"model_{i+1}" for i in range(args.num_models)]
    else:
        model_paths = args.model_paths
    print(f"Testing with model paths: {model_paths}")
    test(url, model_paths, args.slo)
    print("Done.")
