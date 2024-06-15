
# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
# Importing the T5 modules from huggingface/transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments



import utils
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


class YourDataSetClass(Dataset):

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

def my_data_collator(data):
    inputs = []
    targets = []
    for i in data:
        inputs = i[0]
        targets = i[1]
    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def validate(epoch, tokenizer, model, device, loader):

    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        input_batchs = []
        target_batchs = []
        loader_list = loader.values.tolist()
        for num,i in enumerate(loader_list[::10]):
            i_batch = []
            t_batch = []
            for j in range(10):
                if num*10+j >= len(loader_list):
                    break
                i_batch.append(loader_list[num*10+j][0])
                t_batch.append(loader_list[num*10+j][1])
            input_batchs.append(i_batch)
            target_batchs.append(t_batch)

        for input_batch,target_batch in zip(input_batchs,target_batchs):
            ids = tokenizer(
                input_batch,
                max_length=2048,
                return_tensors="pt",
                padding=True)['input_ids'].to(device)
            target = target_batch


            generated_ids = model.generate(
                input_ids = ids,
                max_length=256
                )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            #   target = [tokenizer.batch_decode(t, skip_special_tokens=True)for t in y]


            predictions.extend(preds)
            actuals.extend(target)

    def accuracy_score(outputs, ground_truths):
        correct = 0
        total = 0
        for output, truth in zip(outputs, ground_truths):
            if output.strip().lower().replace(".", "") == truth.strip().lower().replace(".", ""):
                correct += 1
            total += 1
        return correct / total * 100    
    preds = accuracy_score(predictions, actuals)
    print(f'preds:{preds}')
    return predictions, actuals, preds


def T5Trainer(
    dataframe, source_text, target_text, model_params, output_dir="~/resourses/results"
):


    tokenizer = AutoTokenizer.from_pretrained(model_params["MODEL"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_params["MODEL"])
    model = model.to(device)



    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]

    train_dataset = dataframe.iloc[:5]
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)

    train_dataset = train_dataset.reset_index(drop=True)

    val_loader = val_dataset

    # Defining the optimizer that will be used to tune the weights of the network in the training session.

    set_train_dataset = []
    for i in train_dataset.iloc():
        strs = (i['text'],i['headlines'])
        set_train_dataset.append(strs)
    # Training loop
    model.train()
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=model_params["TRAIN_EPOCHS"],
            per_device_train_batch_size=model_params["TRAIN_BATCH_SIZE"],
            per_device_eval_batch_size=model_params["VALID_BATCH_SIZE"],
            weight_decay=0,
            learning_rate=model_params["LEARNING_RATE"],
            evaluation_strategy="no",
            save_total_limit=0,
            eval_steps=None,
            logging_steps=10,
            save_steps=10000,
            logging_dir=output_dir,
        ),
        train_dataset=set_train_dataset,
        data_collator=my_data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()


    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals, preds = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    return preds

import json
task_accs = {}
for path in os.listdir('~/resourses/lora_lab'):
    task_name = path.split('/')[-1].split('.')[0]
    defult_path = path
    with open(path,'r') as f:
        data = json.load(f)
    inputs = []
    outputs = []
    
    for example in data['examples']:
        inputs.append(example['input'])
        outputs.append(example['target'])
    utils.seed_everything(133)
    inputs,outputs = utils.shuffle_list(inputs, outputs)
    example_num = 5
    model_params = {
        "MODEL": "google/flan-t5-large",  # model_type: t5-large
        "TRAIN_BATCH_SIZE": 5,  # training batch size
        "VALID_BATCH_SIZE": 10,  # validation batch size
        "TRAIN_EPOCHS": 40,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 3e-5,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
    }
    df = pd.DataFrame({'text':inputs,'headlines':outputs})


    df["text"] = df["text"]
    preds = T5Trainer(
        dataframe=df,
        source_text="text",
        target_text="headlines",
        model_params=model_params,
        output_dir=f"~/resourses/results/{task_name}",
    )
    task_accs[task_name] = preds
    print(f'{task_name} accuracy:',preds)
utils.save_results(f'~/results/FFT.result', task_accs)

