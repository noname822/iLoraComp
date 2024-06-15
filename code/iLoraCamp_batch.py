import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import argparse
import os
import sys
from peft import PeftModel, PeftConfig
import algorithm
import json
from sentence_transformers import SentenceTransformer
import utils
embedding_model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

seeds = []
for seed in seeds:

    with open('~/resourses/flan_v2_emb.json','r') as f:
        lora_message = json.load(f)
    candidate_lora_path = lora_message.keys()
    candidate_lora_center = torch.tensor([lora_message[i]['centroid'] for i in lora_message])
    candidate_lora_text = [lora_message[i]['text'] for i in lora_message]
    candidate_lora_embedding = torch.tensor(embedding_model.encode(candidate_lora_text))
    task_accs = {}
    for path in os.listdir('~~/resourses/question_list'):
        utils.seed_everything(seed)
        task_name = path.split('/')[-1].split('.')[0]

        dataset_name = path.split('/')[-1]
        def parse_args():
            parser = argparse.ArgumentParser()
            parser.add_argument('--model_name_or_path', type=str, default='google/flan-t5-large')
            return parser.parse_args()


        args = parse_args()

        import json
        train_path = os.path.expanduser(path)
        with open(train_path,'r') as f:
            data = json.load(f)
        inputs = []
        outputs = []
        for example in data['examples']:
            inputs.append(example['input'])
            outputs.append(example['target'])
        example_num = 5
        inputs, outputs = utils.shuffle_list(inputs, outputs)
        print(inputs[:5])
        example_inputs = inputs[0:example_num]
        example_outputs = outputs[0:example_num]
        example_embeddings = embedding_model.encode([i+' '+j for i,j in zip(example_inputs,example_outputs)])
        centroid = torch.mean(torch.tensor(example_embeddings),dim=0)
        #top 20 same lora
        lora_index = torch.argsort(torch.cosine_similarity(candidate_lora_embedding, centroid,dim=1))[:20]
        lora_list = [list(candidate_lora_path)[i] for i in lora_index]
        # print(lora_index)
        model, tokenizer, cache, base_model = algorithm.load_base_model_and_lora_modules(lora_list, args.model_name_or_path)

        accs = {}

        t = 40
        prefs = {}
        
        module_weights,model,tokenizer = algorithm.lorahub_learning(model=model,
                                                                    cache=cache,
                                                                    tokenizer=tokenizer,
                                                                    lora_module_list=lora_list,
                                                                    example_inputs=example_inputs,
                                                                    example_outputs=example_outputs,
                                                                    max_inference_step=t,
                                                                    start_weights=0.0,
                                                                    batch_size=5,
                                                                    )
        # model_weight_stat[i] = module_weights
        valid_inputs, valid_outputs = inputs[example_num:], outputs[example_num:]
        
        example_predictions, perf = algorithm.lorahub_inference(example_inputs=valid_inputs,
                                                        model_or_name_path=model,
                                                        tokenizer_or_tokenizer_path=tokenizer,
                                                        batch_size=10,
                                                        # can set as None if you do not have the ground truth
                                                        example_outputs=valid_outputs)

        task_accs[task_name] = perf
        print("task accuracy:", task_accs)
        
    utils.save_results(f'~/resourses/results/seed{seed}_batch_example_center_same_init00_cosin.result', task_accs)

        
