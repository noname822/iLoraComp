import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import argparse
import os
import sys
from peft import PeftModel, PeftConfig
sys.path.append('/root/dir/scripts/lorahub-register')
import algorithm
import json
sys.path.append('/root/dir/scripts/lorahub-register/choose_lroa')
import utils

seed = 133

task_accs = {}
for path in os.listdir('~/resourses/question_list'):
    task_name = path.split('/')[-1].split('.')[0]
    defult_path = path
    dataset_name = defult_path.split('/')[-1]
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name_or_path', type=str, default='google/flan-t5-large')
        parser.add_argument('--train_path', type=str, default=defult_path)
        return parser.parse_args()


    args = parse_args()

    import json
    train_path = os.path.expanduser(args.train_path)
    with open(train_path,'r') as f:
        data = json.load(f)
    inputs = []
    outputs = []
    for example in data['examples']:
        inputs.append(example['input'])
        outputs.append(example['target'])
    utils.seed_everything(133)

    inputs, outputs = utils.shuffle_list(inputs, outputs)
    example_num = 5
    # random select example_num examples

    example_inputs = inputs[0:example_num]
    example_outputs = outputs[0:example_num]
    #top 20 same lora
    # print(lora_index)
    example = ''
    for i,o in zip(example_inputs,example_outputs):
        example += i+' '+o+'\n'

    accs = {}
    model, tokenizer,_,_ = algorithm.load_base_model_and_lora_modules(model_name_or_path=args.model_name_or_path)
        
    valid_inputs, valid_outputs = [example+x for x in inputs[example_num:]], outputs[example_num:]

    example_predictions, perf = algorithm.lorahub_inference(example_inputs=valid_inputs,
                                                    model_or_name_path=model,
                                                    tokenizer_or_tokenizer_path=tokenizer,
                                                    batch_size=10,
                                                    # can set as None if you do not have the ground truth
                                                    example_outputs=valid_outputs)


    task_accs[task_name] = perf
    
    print("task accuracy:", task_accs)
    
utils.save_results(f'/root/dir/scripts/lorahub-register/choose_lroa/results_random_shuffle/seed{seed}_ICL.result', task_accs)

        
