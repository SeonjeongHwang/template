import os
import sys
import random
import json
import tqdm
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup

from tool.process import *

MIN_FLOAT = -1e30

import argparse

parser = argparse.ArgumentParser(description="EDD")

### Arguments for Traning
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--learning-rate", type=float)
parser.add_argument("--warmup-prop", type=float)

### Arguments for Dataset
parser.add_argument("--version", type=int, default=0)
parser.add_argument("--num-turn", type=int)
parser.add_argument("--max-encoder-length", type=int, default=512)
parser.add_argument("--max-rationale-length", type=int, default=192)
parser.add_argument("--max-answer-length", type=int, default=128)

parser.add_argument("--model-name", type=str, default="t5-large")

### Directories
parser.add_argument("--output-dir", type=str)
parser.add_argument("--train-tag", type=str)

args = parser.parse_args()

if args.version == 0:
    print("Use default input format")
    from tool.data_process import *
elif args.version == 1:
    print("Use highlighted rationale format")
    from tool.data_process_highlight import *

train_data_file="data/coqa/coqa-train-v1.0.json"
train_feature_file=args.output_dir+"/train_features.pkl"
val_data_file="data/coqa/coqa-dev-v1.0.json"
val_feature_file=args.output_dir+"/dev_features.pkl"
test_feature_file=args.output_dir+"/test_features.pkl"

exp_dir = args.output_dir + "/" + args.train_tag
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(exp_dir, exist_ok=True)

model_dir=exp_dir+"/model"
finetuned_model='model/model.pth'
tokenizer_dir=exp_dir+"/tokenizer"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

config = exp_dir+"/config.json"
config_items = {"model_name": args.model_name,
                "max_encoder_length": args.max_encoder_length,
                "max_answer_length": args.max_answer_length,
                "num_turn": args.num_turn,
                "finetuned_model": finetuned_model}

print(f"Check the new config file! {config}")
with open(config, "w") as f:
    json.dump(config_items, f, indent=1)

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 2021
seed_everything(seed)

class Dataset(Dataset):
    def __init__(self, data_file, feature_file, tokenizer, mode):
        self.features = None
        if os.path.exists(feature_file):
            print(f"Loading {mode} features from {feature_file}...")
            with open(feature_file, "rb") as f:
                self.features = pickle.load(f)
        else:
            print(f"Generating {mode} examples...")
            examples = read_examples(input_file=data_file, num_turn=args.num_turn)
            if mode == "train":
                np.random.shuffle(examples)
                total_len = len(examples)
                train_len = int(total_len*0.9)
                train_examples = examples[:train_len]
                test_examples = examples[train_len:]
                print(f"Generating {mode} features...")
                self.features = convert_examples_to_features(examples=train_examples, 
                                                             tokenizer=tokenizer, 
                                                             max_encoder_length=args.max_encoder_length,
                                                             max_rationale_length=args.max_rationale_length,
                                                             max_answer_length=args.max_answer_length)
                print(f"Save the features to {feature_file}...")
                with open(feature_file, "wb") as f:
                    pickle.dump(self.features, f, pickle.HIGHEST_PROTOCOL)
                
                print(f"Generating test features...")
                test_features = convert_examples_to_features(examples=test_examples, 
                                                             tokenizer=tokenizer, 
                                                             max_encoder_length=args.max_encoder_length,
                                                             max_rationale_length=args.max_rationale_length,
                                                             max_answer_length=args.max_answer_length)
                print(f"Save the features to {test_feature_file}...")
                with open(test_feature_file, "wb") as f:
                    pickle.dump(test_features, f, pickle.HIGHEST_PROTOCOL)
                
            else:
                print(f"Generating {mode} features...")
                self.features = convert_examples_to_features(examples=examples, 
                                                             tokenizer=tokenizer, 
                                                             max_encoder_length=args.max_encoder_length,
                                                             max_rationale_length=args.max_rationale_length,
                                                             max_answer_length=args.max_answer_length)
                print(f"Save the features to {feature_file}...")
                with open(feature_file, "wb") as f:
                    pickle.dump(self.features, f, pickle.HIGHEST_PROTOCOL)
            
        self.encoder_input_ids = self.features["encoder_input_ids"]
        self.encoder_attention_mask = self.features["encoder_attention_mask"]
        self.label = self.features["label"]
        
        assert len(self.encoder_input_ids) == len(self.encoder_attention_mask)
        assert len(self.encoder_input_ids) == len(self.label)
        
    def __len__(self):
        return len(self.encoder_input_ids)

    def __getitem__(self, idx):
        encoder_input_ids = torch.tensor(self.encoder_input_ids[idx])
        encoder_attention_mask = torch.tensor(self.encoder_attention_mask[idx])
        label = torch.tensor(self.label[idx])
        
        return encoder_input_ids, encoder_attention_mask, label

class Model(nn.Module):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        self.T5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.T5.resize_token_embeddings(len(tokenizer))
        
    def forward(self, encoder_input_ids, encoder_attention_mask, labels):
        outputs = self.T5(input_ids=encoder_input_ids,
                          attention_mask=encoder_attention_mask,
                          labels=labels)
        
        return outputs
    
def fit(model, train_dataset, val_dataset, device, epochs=2, batch_size=12, warmup_prop=0, lr=3e-5):
    progress_bar = tqdm.tqdm
    model = model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = epochs * len(train_loader)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        
    dev_ppl = 10000000000000000000
    
    for epoch in range(epochs):
        train_pbar = progress_bar(train_loader, total=len(train_loader))
        model.train()
        
        optimizer.zero_grad()
        avg_loss = 0
        
        for encoder_input_ids, encoder_attention_mask, labels in train_pbar:
            outputs = model(encoder_input_ids.to(device),
                            encoder_attention_mask.to(device),
                            labels.to(device))
            
            loss = outputs.loss
            loss.backward()
            
            ppl = torch.exp(loss)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()
            
            train_pbar.set_postfix(loss=float(loss), perplexity=float(ppl))
            
        val_pbar = progress_bar(val_loader, total=len(val_loader))
        model.eval()
        with torch.no_grad():
            total_ppl = []
            for encoder_input_ids, encoder_attention_mask, labels in val_pbar:
                outputs = model(encoder_input_ids.to(device), 
                                encoder_attention_mask.to(device),
                                labels.to(device))
                 
                loss = outputs.loss
                
                ppl = torch.exp(loss)
                
                val_pbar.set_postfix(loss=float(loss), perplexity=float(ppl))
                
                total_ppl.append(float(ppl))
                
            print(f"Epoch {epoch} Validation Perplexity: {sum(total_ppl)/len(total_ppl)}")
            
            if dev_ppl < sum(total_ppl)/len(total_ppl):
                print("Early Stop")
                break
            dev_ppl = sum(total_ppl)/len(total_ppl)   
                
        print(f"Save the model to {os.path.join(exp_dir, finetuned_model)}")
        torch.save(model.state_dict(), os.path.join(exp_dir, finetuned_model))
        
tokenizer = T5Tokenizer.from_pretrained(args.model_name)
if args.version == 1:
    special_tokens_dict = {'additional_special_tokens':['<RH>', '</RH>']}
    tokenizer.add_special_tokens(special_tokens_dict)

print(f"Save the tokenizer into {tokenizer_dir}...")
tokenizer.save_pretrained(tokenizer_dir)

train_dataset = Dataset(data_file=train_data_file, 
                          feature_file=train_feature_file, 
                          tokenizer=tokenizer, 
                          mode="train")
val_dataset = Dataset(data_file=val_data_file, 
                          feature_file=val_feature_file, 
                          tokenizer=tokenizer,
                          mode="val")

model = Model(args.model_name, tokenizer)

device = torch.device("cuda")
fit(model, train_dataset, val_dataset, device, 
    epochs=args.epochs, batch_size=args.batch_size, warmup_prop=args.warmup_prop, lr=args.learning_rate)

print("Done")