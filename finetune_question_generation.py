import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import wandb
import random
from trl import DataCollatorForCompletionOnlyLM
from prepare_datasets import prepare_qgen_qans_with_explanations, prepare_qgen_qans_dgen_with_explanations, prepare_qgen_qans, prepare_qgen_qans_dgen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

new_model_name = "llama3.1_8b_qall_without_explanations"
model_name = "meta-llama/Llama-3.1-8B-Instruct"

wandb.login(key='TOKEN_KEY')
wandb.init(
    project="QAll",
    config={
        'base_model': model_name,
        'model_name': new_model_name,
    }
)

d1 = prepare_qgen_qans_with_explanations()
print(len(d1['train']))
d2 = prepare_qgen_qans_dgen_with_explanations()
print(len(d2["train"]))

data_train = d1['train'] + d2['train']
data_test = d1['val'] + d2['val']

random.shuffle(data_train)
random.shuffle(data_test)

# Convert to dataset
train_dataset = Dataset.from_list(data_train)
test_dataset = Dataset.from_list(data_test)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token='TOKEN_KEY')
tokenizer.pad_token_id = 128002

def tokenize_function(examples):
    messages = [
        {"role": "system", "content": "You are an educational expert."},
        {"role": "user", "content": examples["prompt"]},
        {"role": "assistant", "content": examples["generated_prompt"]},
    ]
    text_example = tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False)
    inputs = tokenizer(text_example, return_tensors="pt", max_length=2048*2, truncation=True, padding=True).to(device)
    inputs["attention_mask"] = inputs["attention_mask"].squeeze()
    inputs["input_ids"] = inputs["input_ids"].squeeze()
    inputs["no_tokens"] = inputs["input_ids"].shape[0]

    return inputs

train_dataset_tokenized = train_dataset.map(lambda x: tokenize_function(x))
test_dataset_tokenized = test_dataset.map(lambda x: tokenize_function(x))

print(len(train_dataset_tokenized))
print(len(test_dataset_tokenized))

# Filter out examples that are too long
train_dataset_tokenized = train_dataset_tokenized.filter(lambda x: x['no_tokens'] <= 4000)
test_dataset_tokenized = test_dataset_tokenized.filter(lambda x: x['no_tokens'] <= 4000)

print(len(train_dataset_tokenized))
print(len(test_dataset_tokenized))

# Drop all columns except input_ids, attention_mask and labels
train_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
test_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])


model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token='TOKEN_KEY')

trainer = Trainer(
    model=model,
    train_dataset=train_dataset_tokenized,
    eval_dataset=test_dataset_tokenized,
    tokenizer=tokenizer,
    args=TrainingArguments(
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        bf16=True,
        dataloader_num_workers=128,
        num_train_epochs=1,
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        optim="adamw_8bit",
        report_to="wandb",
        output_dir=f"logs_{new_model_name}",
        save_steps=0.2,
        save_total_limit=1,
    ),
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False, return_tensors="pt", response_template="<|start_header_id|>assistant<|end_header_id|>")
)

trainer.train()

# Save trained model
trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)
