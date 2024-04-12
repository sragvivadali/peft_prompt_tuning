from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer, LlamaForCausalLM
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
# from torch.cuda.amp import autocast
import pdb 
from mse_distance import compute_mse_from_str
from accelerate import Accelerator
import numpy as np 
from peft import LoraConfig
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
        "--peft", type=str, default='PTuning', help="PTuning | LoRA"
    )
parser.add_argument(
        "--model", type=str, default='7b', help="7b | 13b "
    )
parser.add_argument(
        "--query", type=str, default="imputation", help="imputation | extrapolation"
    )
args = parser.parse_args()

accelerator = Accelerator()

model_name_or_path = f"meta-llama/Llama-2-{args.model}-hf"
tokenizer_name_or_path = f"meta-llama/Llama-2-{args.model}-hf"

assert args.peft in ('PTuning', 'LoRA')
if args.peft == 'PTuning':
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text="Predict the missing values in an ECG signal",
        tokenizer_name_or_path=model_name_or_path,
    )
else:
    #If only targeting attention blocks of the model
    target_modules = ["q_proj", "v_proj"]
    peft_config = LoraConfig(
            r=16,
            target_modules = target_modules,
            lora_alpha=8,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM")

dataset_name = args.query
checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
    "/", "_"
)
text_column = "corrupted ecg"
label_column = "gt values"
max_length = 1400
lr = 3e-2
num_epochs = 20
batch_size = 2

if args.query == 'imputation':
    tr_file, eval_file, ts_file = "./benchmark/ecg_imp_train2.csv", "./benchmark/ecg_imp_eval2.csv", "./benchmark/ecg_imp_test2.csv"
if args.query == 'extrapolation':
    tr_file, eval_file, ts_file = "./benchmark/ecg_exp_train2.csv", "./benchmark/ecg_exp_eval2.csv", "./benchmark/ecg_exp_test2.csv"

dataset = load_dataset("csv", data_files={"train": tr_file, "eval": eval_file, "test": ts_file})

# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Predicted missing values : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
    max_length_batch = 0
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        max_length_batch = max(max_length_batch, len(model_inputs["input_ids"][i]))
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length_batch - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length_batch - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length_batch - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length_batch])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length_batch])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length_batch])
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["eval"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

def test_preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Predicted missing values : " for x in examples[text_column]]
    model_inputs = tokenizer(inputs)
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
    # pdb.set_trace()
    return model_inputs


test_dataset = dataset["test"].map(
    test_preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

# creating model
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# model
# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

device = accelerator.device
# training and evaluation
model = model.to(device)

model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
)

for epoch in range(num_epochs):
    
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        # with autocast(dtype=torch.bfloat16):
        if True:
            # batch = {k: v.to(device) for k, v in batch.items()}
            #         print(batch)
            #         print(batch["input_ids"].shape)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
        
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
    

optimizer.zero_grad()
del optimizer

model.eval()

mse_list = []
for i in range(len(dataset["test"])):
    with torch.no_grad():

        inputs = tokenizer(f'{text_column} : {dataset["test"][i][text_column]} Predicted missing values : ', return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=512, eos_token_id=tokenizer.eos_token_id
        )
        # print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
        pred_str = outputs[0, inputs["input_ids"].shape[-1]:].detach().cpu().numpy()
        pred_str = tokenizer.decode(pred_str, skip_special_tokens=True)
        target_str = dataset["test"][i][label_column]
        mse = compute_mse_from_str('ecg', pred_str, target_str)
        print(i, mse)
        mse_list.append(mse)

print('MSE sample wise: ', mse_list)
print('AVG MSE: ', np.nanmean(mse_list))

