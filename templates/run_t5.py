# https://www.kaggle.com/theoviel/bert-pytorch-huggingface-starter

import os
import random
import json
import tqdm
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from tool.data_process import *

import argparse

parser = argparse.ArgumentParser(description="CRE")

### Arguments for Traning
parser.add_argument("--gpu-num", type=int, default=1)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--learning-rate", type=float)
parser.add_argument("--warmup-prop", type=float)

### Arguments for Dataset
parser.add_argument("--num-turn", type=int)
parser.add_argument("--max-seq-length", type=int, default=512)
parser.add_argument("--doc-stride", type=int, default=128)
parser.add_argument("--max-history-length", type=int, default=128)
parser.add_argument("--max-rationale-length", type=int, default=192)

parser.add_argument("--model-name", type=str, default="bert-large-uncased")
parser.add_argument("--do-lower-case", type=bool, default=True)

### Directories
parser.add_argument("--output-dir", type=str)

args = parser.parse_args()

seed = 2020

print("epochs:", args.epochs)
print("batch size:", args.batch_size)
print("learning rate:", args.learning_rate)
print("warm-up proportion:", args.warmup_prop)

os.makedirs(args.output_dir, exist_ok=True)

tag = str(args.epochs)+"."+str(args.batch_size*args.gpu_num)+"."+str(args.learning_rate)+"."+str(args.warmup_prop)
exp_dir = args.output_dir + "/" + tag
os.makedirs(exp_dir, exist_ok=True)

train_file = "data/coqa/coqa-train-v1.0.json"
train_feature_file = args.output_dir+"/train_features.pkl"
eval_file="data/coqa/coqa-dev-v1.0.json"
eval_feature_file = args.output_dir+"/dev_features.pkl"
test_examples_file = args.output_dir+"/test_examples.pkl"
test_feature_file = args.output_dir+"/test_features.pkl"

model_dir = exp_dir+"/model"
finetuned_model = 'model/model.pth'
tokenizer_dir = exp_dir+"/tokenizer"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

config = exp_dir+"/config.json"
config_items = {"model_name": args.model_name,
                "max_seq_length": args.max_seq_length,
                "doc_stride": args.doc_stride,
                "max_history_length": args.max_history_length,
                "max_rationale_length": args.max_rationale_length,
                "num_turn": args.num_turn,
                "do_lower_case": args.do_lower_case,
                "finetuned_model": finetuned_model}

print(f"Check the new config file! {config}")
with open(config, "w") as f:
    json.dump(config_items, f, indent=1)

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CreDataset(Dataset):
    def __init__(self, data_file, feature_file, tokenizer, mode):
        self.features = None
        if os.path.exists(feature_file):
            print(f"Loading {mode} features from {feature_file}...")
            self.features = load_features_from_pkl(file_name=feature_file)
        else:
            print(f"Generating {mode} examples...")
            examples = read_cre_examples(input_file=data_file, is_training=True, num_turn=args.num_turn)
            if mode == "train":
                np.random.shuffle(examples)
                total_num = len(examples)
                train_num = int(total_num*0.9)
                train_examples = examples[:train_num]
                test_examples = examples[train_num:]
                print(f"Save test examples to {test_examples_file}...")
                with open(test_examples_file, 'wb') as f:
                    pickle.dump(test_examples, f, pickle.HIGHEST_PROTOCOL)
                
                print(f"Generating {mode} features...")
                self.features = convert_examples_to_cre_features(examples=train_examples,
                                                                 tokenizer=tokenizer,
                                                                 max_seq_length=args.max_seq_length,
                                                                 doc_stride=args.doc_stride,
                                                                 max_history_length=args.max_history_length,
                                                                 is_training=True)
                save_features_as_pkl(self.features, feature_file)
                
                print(f"Generating test features...")
                test_features = convert_examples_to_cre_features(examples=test_examples,
                                                                 tokenizer=tokenizer,
                                                                 max_seq_length=args.max_seq_length,
                                                                 doc_stride=args.doc_stride,
                                                                 max_history_length=args.max_history_length,
                                                                 is_training=False)
                save_features_as_pkl(test_features, test_feature_file)
            
            else:
                print(f"Generating {mode} features...")
                self.features = convert_examples_to_cre_features(examples=examples,
                                                                 tokenizer=tokenizer,
                                                                 max_seq_length=args.max_seq_length,
                                                                 doc_stride=args.doc_stride,
                                                                 max_history_length=args.max_history_length,
                                                                 is_training=True)
                save_features_as_pkl(self.features, feature_file)
            
        self.input_ids = self.features["input_ids"]
        self.segment_ids = self.features["segment_ids"]
        self.attention_masks = self.features["attention_mask"]
        
        self.start_positions = self.features["start_position"]
        self.end_positions = self.features["end_position"]
        
        assert len(self.input_ids) == len(self.segment_ids)
        assert len(self.input_ids) == len(self.attention_masks)
        assert len(self.input_ids) == len(self.start_positions)
        assert len(self.input_ids) == len(self.end_positions)
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_id = torch.tensor(self.input_ids[idx])
        segment_id = torch.tensor(self.segment_ids[idx])
        attention_mask = torch.tensor(self.attention_masks[idx])
        
        start_position = torch.tensor(self.start_positions[idx])
        end_position = torch.tensor(self.end_positions[idx])
        
        return input_id, segment_id, attention_mask, start_position, end_position


class CRE(nn.Module):
    def __init__(self, bert_model_name, tokenizer):
        super().__init__()
        self.BertEncoder = BertModel.from_pretrained(bert_model_name)
        self.BertEncoder.resize_token_embeddings(len(tokenizer))
        
        extra_token_type_embedding = nn.Embedding(num_embeddings=1, embedding_dim=self.BertEncoder.config.hidden_size)
        self.BertEncoder.embeddings.token_type_embeddings.weight.data = torch.cat((self.BertEncoder.embeddings.token_type_embeddings.weight.data, extra_token_type_embedding.weight.data), 0)
        
        self.hidden_size = self.BertEncoder.pooler.dense.out_features
        self.span_layer = nn.Linear(in_features=self.hidden_size, out_features=2, bias=True)
        
    def forward(self, input_ids, segment_ids, attention_masks):
        bert_embedding = self.BertEncoder(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=segment_ids).last_hidden_state
        span_logits = self.span_layer(bert_embedding)
        span_logits = torch.transpose(torch.transpose(span_logits, 0, 2), 1, 2) # [0,1,2] -> [2,1,0] -> [2,0,1]
        unbinded_span_logits = torch.unbind(span_logits, dim=0)
        start_logits, end_logits = unbinded_span_logits[0], unbinded_span_logits[1]
        
        return start_logits, end_logits

def fit(model, train_dataset, val_dataset, device, epochs=2, batch_size=12, warmup_prop=0, lr=3e-5):
    model = model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = epochs * len(train_loader)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    loss_fct = nn.CrossEntropyLoss(reduction='mean').to(device) # or sum
    
    min_loss = float("inf")
    progress_bar = tqdm.tqdm
    for epoch in range(epochs):
        train_pbar = progress_bar(train_loader, total=len(train_loader))
        model.train()
        
        optimizer.zero_grad()
        avg_loss = 0
        
        for input_id, segment_id, attention_mask, start_position, end_position in train_pbar:
            start_logits, end_logits = model(input_id.to(device), segment_id.to(device), attention_mask.to(device))
            
            start_loss = loss_fct(start_logits, start_position.to(device))
            end_loss = loss_fct(end_logits, end_position.to(device))
            total_loss = (start_loss + end_loss) * 0.5
            
            total_loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()
            
            train_pbar.set_postfix(total_loss=float(total_loss), start_loss=float(start_loss), end_loss=float(end_loss))
            
        val_pbar = progress_bar(val_loader, total=len(val_loader))
        model.eval()
        with torch.no_grad():
            eval_losses = []
            for input_id, segment_id, attention_mask, start_position, end_position in val_pbar:
                start_logits, end_logits = model(input_id.to(device), segment_id.to(device), attention_mask.to(device))
                
                start_loss = loss_fct(start_logits, start_position.to(device))
                end_loss = loss_fct(end_logits, end_position.to(device))
                total_loss = (start_loss + end_loss) * 0.5
                
                val_pbar.set_postfix(total_loss=float(total_loss), start_loss=float(start_loss), end_loss=float(end_loss))
                
                eval_losses.append(float(total_loss))
            
            print(f"Epoch {epoch} Validation Loss: {sum(eval_losses)/len(eval_losses)}")
            if min_loss < sum(eval_losses)/len(eval_losses):
                print("Early Stopping")
                break
                
            min_loss = sum(eval_losses)/len(eval_losses)
            print(f"Save the model to {os.path.join(exp_dir, finetuned_model)}")
            torch.save(model.state_dict(), os.path.join(exp_dir, finetuned_model))

seed_everything(seed)
tokenizer = BertTokenizer.from_pretrained(args.model_name)

print(f"Save the tokenizer into {tokenizer_dir}...")
tokenizer.save_pretrained(tokenizer_dir)

train_dataset = CreDataset(data_file=train_file, 
                          feature_file=train_feature_file, 
                          tokenizer=tokenizer, 
                          mode="train")
val_dataset = CreDataset(data_file=eval_file, 
                          feature_file=eval_feature_file, 
                          tokenizer=tokenizer, 
                          mode="val")

model = CRE(args.model_name, tokenizer)

device = torch.device("cuda")
fit(model, train_dataset, val_dataset, device, 
    epochs=args.epochs, batch_size=args.batch_size, warmup_prop=args.warmup_prop, lr=args.learning_rate)

print("Done")