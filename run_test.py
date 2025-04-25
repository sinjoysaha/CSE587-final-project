from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json
import evaluate
from tqdm import tqdm
from utils import *

import os, time

print(torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(torch.cuda.device_count())

load_trained = False  # True  #
print(f"Load trained model: {load_trained}")

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
f1 = evaluate.load("f1")
bert_score_eval = evaluate.load("bertscore")

# Load model
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # "google/gemma-3-1b-it"  # "HuggingFaceTB/SmolLM2-135M-Instruct"   #
print(f"{model_name=}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("--- Loading model ---")
if load_trained:
    OUTPUT_DIR = "./llama-sft-lora-checkpoints"
    model = AutoModelForCausalLM.from_pretrained(
        OUTPUT_DIR,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

print("--- Model loaded ---")

get_messages_func = get_messages[model_name]
extract_assistant_response_func = extract_assistant_response[model_name]
print(model.device)


# Load testset
def load_test_data(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        # lines = f.readlines()
        # print(len(lines))
        return [json.loads(line.strip()) for line in f]


def generate_response(question, max_new_tokens=256):
    # inputs = tokenizer(question, return_tensors="pt").to(model.device)
    chat = get_messages_func(question)
    # print(
    #     tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # )
    formatted_chat_tokenized = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(formatted_chat_tokenized, max_new_tokens=max_new_tokens)
    decoded_text = tokenizer.decode(outputs[0])
    # print("----")
    # print(decoded_text)
    return decoded_text


# def clean_text(text):
#     return text.strip().lower()


def evaluate_model(test_data):
    references = []
    predictions = []

    for idx, entry in enumerate(tqdm(test_data, desc="Evaluating")):
        start_time = time.time()
        question = entry["Research Question"]
        expected = get_output_format(entry["Introduction"], entry["Methodology"])
        generated = generate_response(question)
        response = extract_assistant_response_func(generated)

        print(f"RQ: {question}")
        print("*******")
        print(f"Expected: \n{expected}")
        print("*******")
        print(f"Response: \n{response}")
        print("*******")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # generated = response
        predictions.append(response)
        references.append(expected)
        # if idx > 10:
        # break

    with open(
        f"predictions_from_{'pretrained' if load_trained else 'finetuned'}.jsonl", "w"
    ) as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    # Compute metrics
    bleu_score = bleu.compute(
        predictions=predictions, references=references  # [[ref] for ref in references]
    )
    rouge_score = rouge.compute(predictions=predictions, references=references)
    # f1_score = f1.compute(predictions=predictions, references=references)
    bert_score = bert_score_eval.compute(
        predictions=predictions, references=references, model_type="bert-base-uncased"
    )

    print("\n--- Evaluation Results ---")
    print(f"BLEU: {bleu_score['bleu']:.4f}")
    print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
    # print(f"F1 Score: {f1_score['f1']:.4f}")
    print(f"Precision: {np.mean(bert_score['precision']):.4f}")
    print(f"Recall: {np.mean(bert_score['recall']):.4f}")
    print(f"BERT Score: {np.mean(bert_score['f1']):.4f}")


if __name__ == "__main__":
    testset_path = "pqal_test.jsonl"
    test_data = load_test_data(testset_path)
    evaluate_model(test_data)
