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


base_dir = '/Llama2-finetuning'

# Model from local directory
base_model = base_dir + "/Llama-2-7b-chat-hf"

# Dataset from local directory
guanaco_dataset = base_dir + "/guanaco-llama2-1k"

# Fine-tuned model
new_model = "llama-2-7b-chat-guanaco"


dataset = load_dataset(guanaco_dataset, split="train")

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

if torch.backends.mps.is_available():
    print("Using 'mps' (Apple Silicon)")
    active_device = torch.device('mps')
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=active_device
    )
elif torch.cuda.is_available():
    print("Using GPU")
    active_device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map=active_device
    )
else:
    print("Using CPU")
    active_device = torch.device('cpu')
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map=active_device
    )

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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

trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

logging.set_verbosity(logging.CRITICAL)

prompt = "Who is Leonardo Da Vinci?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
