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
from transformers.modeling_outputs import BaseModelOutput

from tool.process import *

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

### Arguments for Inference
parser.add_argument("--top-p", type=float)
parser.add_argument("--top-k", type=int)
parser.add_argument("--temper", type=float)

args = parser.parse_args()

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
    print("Use Rationale")
elif args.version == 7:
    from tool.data_process_g import *
    print("Use Rationale, AH RH, yns")

train_tag = str(args.epochs)+"."+str(args.batch_size*args.gpu_num)+"."+str(args.learning_rate)+"."+str(args.warmup_prop)
exp_dir=args.output_dir+"/"+train_tag

test_tag = str(args.top_p)+"."+str(args.top_k)+"."+str(args.temper)
os.makedirs(exp_dir+"/"+test_tag, exist_ok=True)

model_dir=exp_dir+"/model"
model_type = os.listdir(model_dir)
if "model.pth" in model_type:
    model_file=model_dir+"/model.pth"
elif "pytorch_model.bin" in model_type:
    model_file=model_dir+"/pytorch_model.bin"
else:
    print("There is no pretrained model file!!")
    assert False
tokenizer_dir=exp_dir+"/tokenizer"

test_data_file="data/coqa/coqa-dev-v1.0.json"
test_feature_file=args.output_dir+"/test_features.pkl"

inference_result_file=exp_dir+"/"+test_tag+"/result.json"

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

def generate_one_question(model, encoder_input_ids, encoder_attention_mask, decoder_H_input_ids, decoder_H_attention_mask, decoder_Q_input_ids, decoder_Q_attention_mask, tokenizer, device):
    
    encoder_input_ids = torch.tensor([encoder_input_ids]).to(device)
    encoder_attention_mask = torch.tensor([encoder_attention_mask]).to(device)
    decoder_H_input_ids = torch.tensor([decoder_H_input_ids]).to(device)
    decoder_H_attention_mask = torch.tensor([decoder_H_attention_mask]).to(device)
    decoder_Q_input_ids = torch.tensor([decoder_Q_input_ids]).to(device)
    decoder_Q_attention_mask = torch.tensor([decoder_Q_attention_mask]).to(device)
    
    outputs_H = model.Encoder_Decoder_H(input_ids=encoder_input_ids,
                                        attention_mask=encoder_attention_mask,
                                        decoder_input_ids=decoder_H_input_ids,
                                        decoder_attention_mask=decoder_H_attention_mask)
    
    encoder_outputs = BaseModelOutput(
                      last_hidden_state=outputs_H.encoder_last_hidden_state,
                      hidden_states=outputs_H.encoder_hidden_states,
                      attentions=outputs_H.encoder_attentions,
                      )
    
    decoder_h_kv = outputs_H.past_key_values
    
    generated = model.Decoder_Q.generate(decoder_input_ids=decoder_Q_input_ids,
                                         decoder_attention_mask=decoder_Q_attention_mask,
                                         encoder_outputs=encoder_outputs,
                                         past_key_values=decoder_h_kv,
                                         past_attention_mask=decoder_H_attention_mask,
                                         do_sample=True,
                                         max_length=64, 
                                         top_p=args.top_p, 
                                         top_k=args.top_k, 
                                         temperature=args.temper)
    
    if args.version == 5:
        q_token_id = tokenizer.encode('<Q>')[0]
        output_tokens = []
        start = False
        answer_type = None
        for token in generated[0]:
            if int(token) == q_token_id:
                start = True
                answer_type = ""
                continue
            elif int(token) == tokenizer.eos_token_id:
                break
            if start:
                output_tokens.append(token)
    else:
        yes_token_id = tokenizer.encode('<YES>')[0]
        no_token_id = tokenizer.encode('<NO>')[0]
        span_token_id = tokenizer.encode('<SPAN>')[0]
        output_tokens = []
        start = False
        answer_type = None
        for token in generated[0]:
            if int(token) in [yes_token_id, no_token_id, span_token_id]:
                start = True
                if int(token) == yes_token_id:
                    answer_type = "yes"
                elif int(token) == no_token_id:
                    answer_type = "no"
                else:
                    answer_type = "span"
                continue
            elif int(token) == tokenizer.eos_token_id:
                break
            if start:
                output_tokens.append(token)
        
    return tokenizer.decode(output_tokens), answer_type

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

device = torch.device("cuda")

print("Loading tokenizers...")
tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)

print("Loading the EDD model...")
model = CQG(args.model_name, tokenizer)
model.load_state_dict(torch.load(model_file))
model.eval()

model = model.to(device)

if os.path.exists(test_feature_file):
    print(f"Loading test features from {test_feature_file}...")
    test_features = load_features_from_pkl(file_name=test_feature_file)
else:
    print(f"Generating test examples...")
    test_examples = read_examples(input_file=test_data_file, num_turn=args.num_turn)
    print(f"Generating test features...")
    test_features = convert_examples_to_features(examples=test_examples, 
                                                 tokenizer=tokenizer, 
                                                 encoder_max_length=args.encoder_max_length, 
                                                 decoder_H_max_length=args.decoder_H_max_length, 
                                                 decoder_Q_max_length=args.decoder_Q_max_length,
                                                 is_training=False)
    print(f"Save the features to {test_feature_file}...")
    save_features_as_pkl(test_features, test_feature_file)

encoder_input_ids = test_features["encoder_input_ids"]
encoder_attention_mask = test_features["encoder_attention_mask"]
decoder_H_input_ids = test_features["decoder_H_input_ids"]
decoder_H_attention_mask = test_features["decoder_H_attention_mask"]
decoder_Q_input_ids = test_features["decoder_Q_input_ids"]
decoder_Q_attention_mask = test_features["decoder_Q_attention_mask"]
labels = test_features["label"]

with open(inference_result_file, "w") as writer:
    results = []
    for encoder_input_ids, encoder_attention_mask, decoder_H_input_ids, decoder_H_attention_mask, decoder_Q_input_ids, decoder_Q_attention_mask, label in tqdm.tqdm(zip(encoder_input_ids, encoder_attention_mask, decoder_H_input_ids, decoder_H_attention_mask, decoder_Q_input_ids, decoder_Q_attention_mask, labels), total=len(encoder_input_ids)):
        result, answer_type = generate_one_question(model, encoder_input_ids, encoder_attention_mask, decoder_H_input_ids, decoder_H_attention_mask, decoder_Q_input_ids, decoder_Q_attention_mask, tokenizer, device)
        results.append({"answer": answer_type, "result": result, "label": label})
    json.dump(results, writer, indent=4)
        
print("Done")
