import os
import sys
import random
import json
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from transformers import T5Tokenizer, T5ForConditionalGeneration
from tool.T5_override import T5ForConditionalGeneration as T5ForDecoder
from transformers import AdamW, get_linear_schedule_with_warmup

from tool.process import *

MIN_FLOAT = -1e30

import argparse

parser = argparse.ArgumentParser(description="EDD")

### Arguments for Traning
parser.add_argument("--gpu-num", type=int, default=1)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--learning-rate", type=float)
parser.add_argument("--warmup-prop", type=float)

### Arguments for Dataset
parser.add_argument("--num-turn", type=int)
parser.add_argument("--encoder-max-length", type=int, default=512)
parser.add_argument("--decoder-H-max-length", type=int, default=128)
parser.add_argument("--decoder-Q-max-length", type=int, default=64)

parser.add_argument("--model-name", type=str, default="t5-large")

### Directories
parser.add_argument("--output-dir", type=str)

parser.add_argument("--version", type=int)

args = parser.parse_args()

train_data_file="data/coqa/coqa-train-v1.0.json"
train_feature_file=args.output_dir+"/train_features.pkl"
val_data_file="data/coqa/coqa-dev-v1.0.json"
val_feature_file=args.output_dir+"/dev_features.pkl"
test_feature_file=args.output_dir+"/test_features.pkl"

os.makedirs(args.output_dir, exist_ok=True)

train_tag = str(args.epochs)+"."+str(args.batch_size*args.gpu_num)+"."+str(args.learning_rate)+"."+str(args.warmup_prop)
exp_dir = args.output_dir+"/"+train_tag
os.makedirs(exp_dir, exist_ok=True)

model_dir=exp_dir+"/model"
finetuned_model = "model/model.pth"
tokenizer_dir=exp_dir+"/tokenizer"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

config = exp_dir+"/config.json"
config_items = {"model_name": args.model_name,
                "max_encoder_length": args.encoder_max_length,
                "max_decoder_H_length": args.decoder_H_max_length,
                "max_decoder_Q_length": args.decoder_Q_max_length,
                "num_turn": args.num_turn,
                "finetuned_model": finetuned_model}

print(f"Check the new config file! {config}")
with open(config, "w") as f:
    json.dump(config_items, f, indent=1)

if args.version == 0:
    from tool.data_process import *
    print("Rationale highlight and Yes, No, Span")
elif args.version == 1:
    from tool.data_process_1 import *
    print("Yes, No answer encoding in the Encoder")
elif args.version == 2:
    from tool.data_process_2 import *
    print("history encoder in the Encoder")
elif args.version == 21:
    from tool.data_process_2_1 import *
    print("history encoder in the Encoder & use rationale")
elif args.version == 3:
    from tool.data_process_3 import *
    print("reverse order of history in Decoder-H")
elif args.version == 4:
    from tool.data_process_4 import *
    print("No special tokens in Decoder-H")
elif args.version == 5:
    from tool.data_process_5 import *
    print("Use Rationale")
elif args.version == 6:
    from tool.data_process_6 import *
    print("Use Rationale dividing y,n,s")
elif args.version == 7:
    from tool.data_process_g import *
    print("Use Rationale, AH RH, yns")

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 2020
seed_everything(seed)

class Dataset(Dataset):
    def __init__(self, data_file, feature_file, tokenizer, mode):
        self.features = None
        if os.path.exists(feature_file):
            print(f"Loading {mode} features from {feature_file}...")
            self.features = load_features_from_pkl(file_name=feature_file)
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
                                                             encoder_max_length=args.encoder_max_length, 
                                                             decoder_H_max_length=args.decoder_H_max_length, 
                                                             decoder_Q_max_length=args.decoder_Q_max_length,
                                                             is_training=True)
                print(f"Save the features to {feature_file}...")
                save_features_as_pkl(self.features, feature_file)
                
                print(f"Generating test features...")
                test_features = convert_examples_to_features(examples=test_examples, 
                                                             tokenizer=tokenizer, 
                                                             encoder_max_length=args.encoder_max_length, 
                                                             decoder_H_max_length=args.decoder_H_max_length, 
                                                             decoder_Q_max_length=args.decoder_Q_max_length,
                                                             is_training=False)
                print(f"Save the features to {test_feature_file}...")
                save_features_as_pkl(test_features, test_feature_file)
                
            else:
                print(f"Generating {mode} features...")
                self.features = convert_examples_to_features(examples=examples, 
                                                             tokenizer=tokenizer, 
                                                             encoder_max_length=args.encoder_max_length, 
                                                             decoder_H_max_length=args.decoder_H_max_length, 
                                                             decoder_Q_max_length=args.decoder_Q_max_length,
                                                             is_training=True)
                print(f"Save the features to {feature_file}...")
                save_features_as_pkl(self.features, feature_file)
            
        self.encoder_input_ids = self.features["encoder_input_ids"]
        self.encoder_attention_mask = self.features["encoder_attention_mask"]
        self.decoder_H_input_ids = self.features["decoder_H_input_ids"]
        self.decoder_H_attention_mask = self.features["decoder_H_attention_mask"]
        self.decoder_Q_input_ids = self.features["decoder_Q_input_ids"]
        self.decoder_Q_attention_mask = self.features["decoder_Q_attention_mask"]
        
        assert len(self.encoder_input_ids) == len(self.encoder_attention_mask)
        assert len(self.encoder_input_ids) == len(self.decoder_H_input_ids)
        assert len(self.encoder_input_ids) == len(self.decoder_H_attention_mask)
        assert len(self.encoder_input_ids) == len(self.decoder_Q_input_ids)
        assert len(self.encoder_input_ids) == len(self.decoder_Q_attention_mask)
        
    def __len__(self):
        return len(self.encoder_input_ids)

    def __getitem__(self, idx):
        encoder_input_ids = torch.tensor(self.encoder_input_ids[idx])
        encoder_attention_mask = torch.tensor(self.encoder_attention_mask[idx])
        
        decoder_H_input_ids = torch.tensor(self.decoder_H_input_ids[idx])
        decoder_H_attention_mask = torch.tensor(self.decoder_H_attention_mask[idx])
        decoder_Q_input_ids = torch.tensor(self.decoder_Q_input_ids[idx])
        decoder_Q_attention_mask = torch.tensor(self.decoder_Q_attention_mask[idx])
        
        return encoder_input_ids, encoder_attention_mask, decoder_H_input_ids, decoder_H_attention_mask, decoder_Q_input_ids, decoder_Q_attention_mask

class CQG(nn.Module):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        self.Encoder_Decoder_H = T5ForConditionalGeneration.from_pretrained(model_name)
        self.Encoder_Decoder_H.resize_token_embeddings(len(tokenizer))
        self.Decoder_Q = T5ForDecoder.from_pretrained(model_name)
        self.Decoder_Q.resize_token_embeddings(len(tokenizer))
        del self.Decoder_Q.encoder
        
    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_H_input_ids, decoder_H_attention_mask, decoder_Q_input_ids, decoder_Q_attention_mask):
        outputs_H = self.Encoder_Decoder_H(input_ids=encoder_input_ids,
                                         attention_mask=encoder_attention_mask,
                                         decoder_input_ids=decoder_H_input_ids,
                                         decoder_attention_mask=decoder_H_attention_mask)
        encoder_outputs = [outputs_H.encoder_last_hidden_state, 
                           outputs_H.encoder_hidden_states, 
                           outputs_H.encoder_attentions]
        decoder_h_kv = outputs_H.past_key_values
        
        outputs_Q = self.Decoder_Q(decoder_input_ids=decoder_Q_input_ids,
                                   decoder_attention_mask=decoder_Q_attention_mask,
                                   encoder_outputs=encoder_outputs,
                                   past_key_values=decoder_h_kv,
                                   past_attention_mask=decoder_H_attention_mask)
        return outputs_Q.logits
    
def fit(model, train_dataset, val_dataset, device, epochs=2, batch_size=12, warmup_prop=0, lr=3e-5):
    progress_bar = tqdm.tqdm
    model = model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = epochs * len(train_loader)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    loss_fct = nn.CrossEntropyLoss().to(device)
    
    dev_ppl = 10000000000000000000
    
    for epoch in range(epochs):
        train_pbar = progress_bar(train_loader, total=len(train_loader))
        model.train()
        
        optimizer.zero_grad()
        avg_loss = 0
        
        for encoder_input_ids, encoder_attention_mask, decoder_H_input_ids, decoder_H_attention_mask, decoder_Q_input_ids, decoder_Q_attention_mask in train_pbar:
            logits = model(encoder_input_ids.to(device), 
                           encoder_attention_mask.to(device), 
                           decoder_H_input_ids.to(device), 
                           decoder_H_attention_mask.to(device), 
                           decoder_Q_input_ids.to(device), 
                           decoder_Q_attention_mask.to(device))
            
            shift_logits = logits[..., :-1, :].contiguous().to(device)
            shift_labels = decoder_Q_input_ids[..., 1:].contiguous().to(device)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
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
            for encoder_input_ids, encoder_attention_mask, decoder_H_input_ids, decoder_H_attention_mask, decoder_Q_input_ids, decoder_Q_attention_mask in val_pbar:
                logits = model(encoder_input_ids.to(device), 
                           encoder_attention_mask.to(device), 
                           decoder_H_input_ids.to(device), 
                           decoder_H_attention_mask.to(device), 
                           decoder_Q_input_ids.to(device), 
                           decoder_Q_attention_mask.to(device))
                 
                shift_logits = logits[..., :-1, :].contiguous().to(device)
                shift_labels = decoder_Q_input_ids[..., 1:].contiguous().to(device)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  
                
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
special_tokens_dict = {'sep_token': '<SEP>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'additional_special_tokens':['<AH>', '</AH>', '<RH>', '</RH>', '<YES>', '<NO>', '<SPAN>', '<Q>', '<A>']}
if args.version == 5:
    special_tokens_dict = {'sep_token': '<SEP>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'additional_special_tokens':['<HR>', '</HR>', '<Q>', '<A>']}
if args.version in [0,6]:
    special_tokens_dict = {'sep_token': '<SEP>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'additional_special_tokens':['<HR>', '</HR>', '<Q>', '<A>', '<YES>', '<NO>', '<SPAN>']}
tokenizer.add_special_tokens(special_tokens_dict)

print(f"Save the tokenizer into {tokenizer_dir}...")
tokenizer.save_pretrained(tokenizer_dir)

val_dataset = Dataset(data_file=val_data_file, 
                          feature_file=val_feature_file, 
                          tokenizer=tokenizer,
                          mode="val")
train_dataset = Dataset(data_file=train_data_file, 
                          feature_file=train_feature_file, 
                          tokenizer=tokenizer, 
                          mode="train")


model = CQG(args.model_name, tokenizer)

device = torch.device("cuda")
fit(model, train_dataset, val_dataset, device, 
    epochs=args.epochs, batch_size=args.batch_size, warmup_prop=args.warmup_prop, lr=args.learning_rate)

print("Done")
