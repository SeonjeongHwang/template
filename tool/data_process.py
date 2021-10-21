import json
import pandas as pd
import tqdm
import copy

from tool.process import *

class Example(object):
    
    def __init__(self,
                 qas_id,
                 target_answer,
                 answer_type,
                 prev_qa,
                 question_text,
                 doc_tokens,
                 start_position,
                 end_position):
        self.qas_id = qas_id
        self.target_answer = target_answer
        self.answer_type = answer_type
        self.prev_qa = prev_qa
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.start_position = start_position
        self.end_position = end_position
    
def read_examples(input_file, num_turn=1):
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
        
    examples = []
    for idx, data in tqdm.tqdm(enumerate(input_data), total=len(input_data)):
        data_id = data["id"]
        document = data["story"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in document:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
            
        questions = sorted(data["questions"], key=lambda x: x["turn_id"])
        answers = sorted(data["answers"], key=lambda x: x["turn_id"])
        
        prev_qa = ""
        prev_qa_list = []
        qas = list(zip(questions, answers))
        for i, (question, answer) in enumerate(qas):
            qas_id = "{0}_{1}".format(data_id, i+1)
            
            answer_type, answer_subtype = get_answer_type(question, answer)
            answer_text, span_start, span_end, is_skipped, rat_text, rat_start, rat_end = get_answer_span(answer, answer_type, document)

            if answer_type in ["unknown"]:
                prev_qa_list.append((question["input_text"], answer["input_text"]))
                continue

            start_position = char_to_word_offset[rat_start]
            end_position = char_to_word_offset[rat_end]
            target_answer = answer_text
                
            example = Example(
                qas_id=qas_id,
                target_answer=target_answer,
                answer_type=answer_type,
                prev_qa = prev_qa_list[-num_turn:].copy(),
                question_text=question["input_text"],
                doc_tokens=doc_tokens,
                start_position=start_position,
                end_position=end_position)
            
            examples.append(example)
            prev_qa_list.append((question["input_text"], answer["input_text"]))
            
    return examples

def convert_examples_to_features(examples, tokenizer, encoder_max_length, decoder_H_max_length, decoder_Q_max_length, fore_space=32, back_space=32, is_training=True):

    if is_training:
        features = {
            "encoder_input_ids": [],
            "encoder_attention_mask": [],
            "decoder_H_input_ids": [],
            "decoder_H_attention_mask": [],
            "decoder_Q_input_ids": [],
            "decoder_Q_attention_mask": [],
        }
    else:
        features = {
            "encoder_input_ids": [],
            "encoder_attention_mask": [],
            "decoder_H_input_ids": [],
            "decoder_H_attention_mask": [],
            "decoder_Q_input_ids": [],
            "decoder_Q_attention_mask": [],
            "label": [],
        }    
    
    """
    context(with <RH> rationale </RH>) <SEP> <A> A <Q> Q <A> A <Q> Q <A> answer
    <Q> Q <A> A <Q> Q <A> A
    <YES,NO,SPAN>
    """
    
    rationale_start_token_id = tokenizer.encode('<RH>')[0]
    rationale_end_token_id = tokenizer.encode('</RH>')[0]
    yes_token_id = tokenizer.encode("<YES>")[0]
    no_token_id = tokenizer.encode("<NO>")[0]
    span_token_id = tokenizer.encode("<SPAN>")[0]
    Q_token_id = tokenizer.encode("<Q>")[0]
    A_token_id = tokenizer.encode("<A>")[0]
    
    for (example_index, example) in tqdm.tqdm(enumerate(examples), total=len(examples)):
        
        ### ENCODER ### context(with a or r highlight) <SEP> <A> <Q> <A> <Q> <A>
        history_input_ids = []
        if len(example.prev_qa) > 0:
            for q, a in example.prev_qa:
                history_input_ids.append(A_token_id)
                history_input_ids += tokenizer.encode(a)[:-1]
                history_input_ids.append(Q_token_id)
                history_input_ids += tokenizer.encode(q)[:-1]
                
        history_max_length = decoder_H_max_length
        if len(history_input_ids) > history_max_length:
            history_input_ids = history_input_ids[-history_max_length:]
            
        history_input_ids += [tokenizer.sep_token_id] + history_input_ids # <SEP> history
        history_input_ids.append(A_token_id)
        history_input_ids += tokenizer.encode(example.target_answer)[:-1] # <SEP> history <A> answer 
        
        encoded_doc_tokens = []
        encoded_start_position = None
        encoded_end_position = None
        for idx, doc_token in enumerate(example.doc_tokens):
            if idx == example.start_position:
                encoded_start_position = len(encoded_doc_tokens)
            encoded_doc_tokens.extend(tokenizer.encode(doc_token)[:-1])
            if idx == example.end_position:
                encoded_end_position = len(encoded_doc_tokens) - 1

        #context_start_position = max(0, encoded_start_position-fore_space)
        context_end_position = min(len(encoded_doc_tokens)-1, encoded_end_position+back_space)
        
        encoder_input_ids = encoded_doc_tokens[:encoded_start_position] + [rationale_start_token_id] + encoded_doc_tokens[encoded_start_position:encoded_end_position+1] + [rationale_end_token_id] + encoded_doc_tokens[encoded_end_position+1:context_end_position+1]
            
        max_doc_length = encoder_max_length - len(history_input_ids)
        
        if len(encoder_input_ids) > max_doc_length:
            encoder_input_ids = encoder_input_ids[-max_doc_length:]
        
        encoder_input_ids = encoder_input_ids + history_input_ids
        encoder_attention_mask = [1]*len(encoder_input_ids)
        
        while len(encoder_input_ids) < encoder_max_length:
            encoder_input_ids.append(tokenizer.pad_token_id)
            encoder_attention_mask.append(0)
            
        assert len(encoder_input_ids) == encoder_max_length
        assert len(encoder_attention_mask) == encoder_max_length
                
        ### DECODER-H ### history
        decoder_H_input_ids = []
        if len(example.prev_qa) > 0:
            for q, a in example.prev_qa:
                decoder_H_input_ids.append(Q_token_id)
                decoder_H_input_ids += tokenizer.encode(q)[:-1]
                decoder_H_input_ids.append(A_token_id)
                decoder_H_input_ids += tokenizer.encode(a)[:-1]
        
        if len(decoder_H_input_ids) > decoder_H_max_length:
            decoder_H_input_ids = decoder_H_input_ids[-decoder_H_max_length:]

        decoder_H_attention_mask = [1]*len(decoder_H_input_ids)
            
        while len(decoder_H_input_ids) < decoder_H_max_length:
            decoder_H_input_ids.append(tokenizer.pad_token_id)
            decoder_H_attention_mask.append(0)
                
        assert len(decoder_H_input_ids) == decoder_H_max_length
        assert len(decoder_H_attention_mask) == decoder_H_max_length
                
        ### DECODER-Q ### <Y or N or S> question
        decoder_Q_input_ids = []
        if example.answer_type == "yes":
            decoder_Q_input_ids += [yes_token_id]
        elif example.answer_type == "no":
            decoder_Q_input_ids += [no_token_id]
        else:
            decoder_Q_input_ids += [span_token_id]
         
        decoder_Q_attention_mask = [1]

        if is_training:
            tokenized_question = tokenizer(example.question_text)
            decoder_Q_input_ids += tokenized_question["input_ids"]
            decoder_Q_attention_mask += tokenized_question["attention_mask"]
        if not is_training:
            label = example.question_text
            
        if len(decoder_Q_input_ids) > decoder_Q_max_length:
            print("Question is too long", len(decoder_Q_input_ids)-1)
            continue

        if is_training:
            while len(decoder_Q_input_ids) < decoder_Q_max_length:
                decoder_Q_input_ids.append(tokenizer.pad_token_id)
                decoder_Q_attention_mask.append(0)
                
        if example_index < 10:
            print("encoder_input_ids:", tokenizer.decode(encoder_input_ids))
            print("encoder_input_ids:", encoder_input_ids)
            print("decoder_H_input_ids:", tokenizer.decode(decoder_H_input_ids))
            print("decoder_H_input_ids:", decoder_H_input_ids)
            print("decoder_Q_input_ids:", tokenizer.decode(decoder_Q_input_ids))
            print("decoder_Q_input_ids:", decoder_Q_input_ids)
            print("encoder_attention_mask:", encoder_attention_mask)
            print("decoder_H_attention_mask:", decoder_H_attention_mask)
            print("decoder_Q_attention_mask:", decoder_Q_attention_mask)
        
        
        features["encoder_input_ids"].append(encoder_input_ids)
        features["decoder_H_input_ids"].append(decoder_H_input_ids)
        features["decoder_Q_input_ids"].append(decoder_Q_input_ids)
               
        features["encoder_attention_mask"].append(encoder_attention_mask)
        features["decoder_H_attention_mask"].append(decoder_H_attention_mask)
        features["decoder_Q_attention_mask"].append(decoder_Q_attention_mask)
        
        if not is_training:
            features["label"].append(label)
    
    return features

def save_features_as_pkl(features, file_name):
    temp = pd.DataFrame(features)
    temp.to_pickle(file_name)
    
def load_features_from_pkl(file_name):
    data = pd.read_pickle(file_name).to_dict()
    features = dict()
    for key, values in data.items():
        features[key] = list(values.values())
    return features