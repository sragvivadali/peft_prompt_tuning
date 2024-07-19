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
from accelerate import Accelerator
import numpy as np 
from peft import LoraConfig
import argparse
from datasets import load_dataset
import matplotlib.pyplot as plt

from mse_distance import compute_mse_from_str, compute_mae
from ecg2csv2 import initialize_files
from plot_and_write import write_to_output, plot_predicted_vs_ground_truth

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft", type=str, default='PTuning', help="PTuning | LoRA")
    parser.add_argument("--model", type=str, default='7b', help="7b | 13b ")
    parser.add_argument("--query", type=str, default="imputation", help="imputation | extrapolation")
    parser.add_argument("-f", "--folder", type=str, help="folder name", default="./benchmark")
    parser.add_argument("-p", "--prompt", type=str, default="Predict the missing values")
    parser.add_argument("-t", "--text", type=str, default="corrupted data")
    parser.add_argument("-l", "--label", type=str, default="gt values")
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument("--eval", type=int, default=50)
    parser.add_argument("--pred", type=int, default=50)
    return parser.parse_args()

def get_peft_config(args, model_name_or_path):
    assert args.peft in ('PTuning', 'LoRA')
    if args.peft == 'PTuning':
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=8,
            prompt_tuning_init_text=args.prompt,
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

    return peft_config

def load_and_preprocess_datasets(dataset, tokenizer, text_column, label_column, max_length, args):
    batch_size = 2

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

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )

    test_dataset = dataset["test"].map(
        test_preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    test_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )

    return train_dataloader, eval_dataloader, test_dataloader

def train_and_evaluate(model, device, tokenizer, optimizer, lr_scheduler, train_dataloader, eval_dataloader, accelerator, num_epochs):
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

    del optimizer
    return model

def test_model(dataset, model, device, test_dataloader, tokenizer, text_column, label_column, args, max_vals):
    model.eval()

    mse_list = []
    mae_list = []
    rmse_list = []

    pred_values = []
    gt_values = []
    validity_count = 0
    current_entry = 0

    def is_integer(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    for batch in test_dataloader:
        for i in range(len(batch["input_ids"])):
            invalid = False
            with torch.no_grad():
                inputs = tokenizer(f'{text_column} : {dataset["test"][current_entry][text_column]} {args.prompt} : ', return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=512, eos_token_id=tokenizer.eos_token_id
                )
            
            pred_str = tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1]:].detach().cpu().numpy(), skip_special_tokens=True)
            target_str = dataset["test"][current_entry][label_column]

            pred_list = []
            for x in pred_str.split(', '):
                if is_integer(x):
                    pred_list.append(float(x))
                else:
                    invalid = True

            pred_list = np.array(pred_list)
    
            gt_list = np.array([float(x) for x in target_str.split(', ') if is_integer(x)])

            if len(pred_list) != len(gt_list) or invalid:
                validity_count += 1

            mse = compute_mse_from_str(max_vals[current_entry], pred_list, gt_list)
            mae = compute_mae(max_vals[current_entry], pred_list, gt_list)
            pred_values.append(pred_list)
            gt_values.append(gt_list)

            print("Maximum Value in Original data: ", max_vals[current_entry])
            print("Predicted String: ", pred_str)
            print("Target String: ", target_str)

            print("MSE: ", mse)
            print("MAE: ", mae)
            print("RMSE: ", mse ** 0.5)

            # maybe also look at (MAPE)

            write_to_output(filename = f"output_for_{args.train}_and_{args.pred}.txt", var_names = ["max value", "pred_str", "target_str", "mse", "mae", "rsme"], values = [max_vals[current_entry], pred_str,target_str, mse, mae, mse ** 0.5])

            mse_list.append(mse)
            mae_list.append(mae)
            rmse_list.append(mse ** 0.5)
            current_entry += 1

    plot_predicted_vs_ground_truth(pred_values, gt_values, title = f"Ground Truth vs Predicted Data", output = f"output_for_{args.train}_and_{args.pred}.png", query = args.query)

    print('AVG MSE: ', np.nanmean(mse_list))
    print('Median MSE: ', np.nanmedian(mse_list))

    print('AVG RMSE: ', np.nanmean(rmse_list))
    print('Median RMSE: ', np.nanmedian(rmse_list))

    print('AVG MAE: ', np.nanmean(mae_list))
    print('Median MAE: ', np.nanmedian(mae_list))

    print('Number of False Outputs: ', validity_count)

    write_to_output(filename = f"output_for_{args.train}_and_{args.pred}.txt", var_names = ["AVG MSE", "Median MSE", "AVG RMSE", "Median RMSE", "AVG MAE", "Median MAE", "Number of False Outputs"], 
    values = [np.nanmean(mse_list), np.nanmedian(mse_list), np.nanmean(rmse_list), np.nanmedian(rmse_list), np.nanmean(mae_list), np.nanmedian(mae_list), validity_count])


def main():
    args = parse_arguments()
    accelerator = Accelerator()

    # Initialize files
    max_vals = initialize_files(args.folder, args.text, args.label, args.train, args.test, args.eval, pred_len=args.pred)

    model_name_or_path = f"meta-llama/Llama-2-{args.model}-hf"
    tokenizer_name_or_path = f"meta-llama/Llama-2-{args.model}-hf"

    assert args.peft in ('PTuning', 'LoRA')
    peft_config = get_peft_config(args, model_name_or_path)

    dataset_name = args.query
    checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
        "/", "_"
    )

    dataset = load_dataset("csv", data_files={
        "train": "./benchmark/train.csv", 
        "eval": "./benchmark/eval.csv", 
        "test": "./benchmark/test.csv"
    })

    # Load and preprocess datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_input_length = max(len(input_text) * 1.1 for input_text in dataset['train'][args.text])
    max_length = int(max_input_length)

    num_epochs = 10

    train_dataloader, eval_dataloader, test_dataloader = load_and_preprocess_datasets(
        dataset, tokenizer, args.text, args.label, max_length, args
    )

    # Create model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    device = accelerator.device
    model = model.to(device)

    # Prepare everything with the accelerator
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    )

    # Training and evaluation
    model = train_and_evaluate(model, device, tokenizer, optimizer, lr_scheduler, train_dataloader, eval_dataloader, accelerator, num_epochs)

    # Testing
    test_model(dataset, model, device, test_dataloader, tokenizer, args.text, args.label, args, max_vals)

if __name__ == "__main__":
    main()