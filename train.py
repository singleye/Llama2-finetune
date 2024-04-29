import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="model to load, eg: NousResearch/Llama-2-7b-chat-hf or /local/path/to/model")
parser.add_argument("-d", "--dataset", required=True, help="dataset to load, eg: mlabonne/guanaco-llama2-1k or /local/path/to/dataset")
parser.add_argument("-s", "--save-model", required=True, help="save the trained model as")

args = parser.parse_args()


base_dir = '/Llama2-finetuning'

# Model from local directory
model_name = args.model
print(f"Model: {model_name}")

# Dataset from local directory
dataset_name = args.dataset
print(f"Dataset: {dataset_name}")

# Fine-tuned model
new_model = args.save_model

dataset = load_dataset(dataset_name, split="train")
print(f"Dataset loaded")

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model_options = {
    'pretrained_model_name_or_path': model_name,
}
if torch.backends.mps.is_available():
    print("Using 'mps' (Apple Silicon)")
    model_options["device_map"] = torch.device('mps')
    model_options["low_cpu_mem_usage"] = True
    model_options["trust_remote_code"] = True
elif torch.cuda.is_available():
    print("Using GPU")
    model_options["device_map"] = torch.device('cuda')
    model_options["quantization_config"] = quant_config
else:
    print("Using CPU")
    active_device = torch.device('cpu')
    model_options["device_map"] = torch.device('cpu')
    model_options["quantization_config"] = quant_config

print(f"Model options: {model_options}")
model = AutoModelForCausalLM.from_pretrained(**model_options)
model.config.use_cache = False
model.config.pretraining_tp = 1
print(f"Model loaded")

print(f"Tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"Tokenizer loaded")

print(f"Prepare for training model: {new_model}")
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
	output_dir="./results",
	num_train_epochs=1,
	per_device_train_batch_size=4,
	gradient_accumulation_steps=1,
	gradient_checkpointing = True,
	learning_rate=2e-4,
	weight_decay=0.001,
	lr_scheduler_type="constant",
	warmup_ratio=0.03,
	max_grad_norm=0.3,
	max_steps=-1,
	save_steps=25,
	logging_steps=25,
	logging_dir="./logs",
	group_by_length=True,
	fp16=False,
	report_to="tensorboard",
	adam_beta2=0.999,
	do_train=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
print("Training started...")
trainer.train()
print("Training finished")

print(f"Save model: {new_model}")
trainer.model.save_pretrained(new_model)
print(f"Save tokenizer: {new_model}")
trainer.tokenizer.save_pretrained(new_model)

#logging.set_verbosity(logging.CRITICAL)
#
#prompt = "Who is Leonardo Da Vinci?"
#pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
#result = pipe(f"<s>[INST] {prompt} [/INST]")
#print(result[0]['generated_text'])
