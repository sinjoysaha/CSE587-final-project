import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM  # , TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import json
import os, time
from utils import *

print(torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.device_count())

# --- CONFIG ---
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # "google/gemma-3-1b-it"  # "google/gemma-2b"  # or "google/gemma-7b", "google/gemma-1.5b"
DATASET_PATH = "pqal_train.jsonl"
OUTPUT_DIR = "./llama-sft-lora-checkpoints"
print(f"{MODEL_NAME=}")
get_messages_prompt_resp_func = get_messages_prompt_resp[MODEL_NAME]

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --- Load 4-bit Model ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
)

# --- Apply LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)


# --- Load Dataset ---
def load_jsonl_dataset(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]

    # return Dataset.from_list(
    #     [
    #         {
    #             "prompt": get_prompt(ex["Research Question"]),
    #             "completion": get_output_format(ex["Introduction"], ex["Methodology"]),
    #         }
    #         for ex in data
    #     ]
    # )
    return Dataset.from_list(
        [
            {
                "messages": get_llama_messages_prompt_resp(
                    ex["Research Question"], ex["Introduction"], ex["Methodology"]
                )
            }
            for ex in data
        ]
    )


dataset = load_jsonl_dataset(DATASET_PATH)
print("--- Dataset loaded ---")
print(dataset)

print("--- SFTConfig loaded ---")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    # dataset_text_field="messages",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    fp16=True,
    bf16=False,
    # packing=True,
    # eval_strategy="no",
    report_to="none",
    # remove_unused_columns=False,
)

print("--- SFTTrainer loaded---")
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
)

print("--- Start training ---")
trainer.train()
print("--- Training finished ---")

print("--- Saving model ---")
trainer.save_model(OUTPUT_DIR)

print("--- Loading model ---")
model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
)
print("--- Model loaded ---")
