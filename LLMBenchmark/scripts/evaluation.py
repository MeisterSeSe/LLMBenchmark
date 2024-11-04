import json
from datetime import datetime
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import gc

models_info = {
    "phi-2": {
        "path": "microsoft/phi-2",
        "params": "2.7B",
        "published": "December 2023 (widely available in 2024)"
    },
    "bloomz-1b7": {
        "path": "bigscience/bloomz-1b7",
        "params": "1.7B",
        "published": "November 2022"
    },
    "stablelm-2-1_6b": {
        "path": "stabilityai/stablelm-2-1_6b",
        "params": "1.6B",
        "published": "January 2024",
        "description": "Stability AI's latest small-scale model, trained on a diverse dataset for general-purpose use."
    },
    "tinyllama-1.1b-chat": {
        "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "params": "1.1B",
        "published": "January 2024",
        "description": "A compact model trained to emulate larger language models, optimized for efficient deployment "
                       "and chat-like interactions."
    },
    "opt-1.3b": {
        "path": "facebook/opt-1.3b",
        "params": "1.3B",
        "published": "May 2022"
    },
}

# Benchmark questions
questions = [
    "List the top 5 AI companies globally with the highest funding in 2023.",
    "Identify the top 3 cloud service providers by market share in 2023 and provide their market share percentages.",
    #https://www.crn.com/news/cloud/2024/cloud-market-share-q4-2023-results-aws-falls-as-microsoft-grows?page=5
    "List the top 5 trending libraries in the Data Science market as of Q4 2023, along with their primary use cases "
    "and growth rates over the past year.",
    "What is the pricing range for JetBrains PyCharm? Which pricing package would you recommend a student?",
    "Using the historical growth rate of the global AI market from 2020 to 2023, estimate the market size for 2025. "
    "Provide your reasoning and state any assumptions."
]


# Evaluation functions
def calculate_average_times(file_name):
    """Calculates the average response time for each model across the questions, needs the benchmark.json file"""
    with open(file_name, 'r') as file:
        data = json.load(file)
    avg_times = []
    for model, responses in data.items():
        times = []
        for question in range(len(responses)):
            times.append(responses[question]['response_time'])
        avg_times.append(sum(times) / len(times))
    return avg_times


# Main evaluation function
def evaluate_response(response):
    return {
        "response_time": response["response_time"],
        "token_efficiency": len(response["content"]) / response["tokens_used"],
    }


# Function to get model response
def get_model_response(model, tokenizer, question):
    # Prepare the prompt
    prompt = f"Question: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

    end_time = time.time()
    response_time = end_time - start_time

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated answer (remove the input prompt)
    answer = response.split("Answer:", 1)[-1].strip()

    return {
        "content": answer,
        "response_time": response_time,
        "tokens_used": len(outputs[0])
    }


# Run benchmark
def run_benchmark(models_info, questions):
    results = {}
    model_responses = {}

    for model_name, model_info in models_info.items():
        print(f"Benchmarking {model_name}...")

        # Load the model
        model, tokenizer = load_single_model(model_name, model_info)

        model_results = []
        model_responses[model_name] = []

        for question in questions:
            response = get_model_response(model, tokenizer, question)
            evaluation = evaluate_response(response)
            model_results.append(evaluation)
            model_responses[model_name].append(response["content"])

        results[model_name] = model_results

        # Unload the model
        unload_model(model)
        print(f"Finished benchmarking {model_name} and unloaded the model.")

        # Print current GPU memory usage
        if torch.cuda.is_available():
            print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Current GPU memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    return results, model_responses


# Unload model from GPU
def unload_model(model):
    model_name = model.__class__.__name__
    print(f"Unloading {model_name}...")

    # Move model to CPU before deletion
    model.cpu()

    # Delete the model
    del model

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run garbage collection
    gc.collect()

    print(f"{model_name} unloaded and memory cleared.")


# Load model from hugging face to GPU if possible
def load_single_model(model_name, model_info):
    print(f"Loading {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_info["path"])

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_info["path"],
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )

    print(f"{model_name} loaded successfully.")
    return model, tokenizer


def save_results(results, model_responses):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save evaluation results
    with open(f'../results/benchmark_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save model responses
    with open(f'results/model_responses_{timestamp}.json', 'w') as f:
        json.dump(model_responses, f, indent=2)

    print(f"Results and responses saved with timestamp {timestamp}")


# Main execution
if __name__ == "__main__":
    results, model_responses = run_benchmark(models_info, questions)
    save_results(results, model_responses)
