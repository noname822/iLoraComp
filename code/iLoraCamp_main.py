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
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='google/flan-t5-large')
    return parser.parse_args()

args = parse_args()
seeds = []
for seed in seeds:
    

    with open('~/resourses/flan_v2_emb.json','r') as f:
        lora_message = json.load(f)
    candidate_lora_path = lora_message.keys()
    candidate_lora_center = torch.tensor([lora_message[i]['centroid'] for i in lora_message])
    candidate_lora_text = [lora_message[i]['text'] for i in lora_message]
    candidate_lora_embedding = candidate_lora_center
    task_accs = {}
    for path in os.listdir('~/resourses/question_list'):
        task_name = path.split('/')[-1].split('.')[0]
        defult_path = path
        dataset_name = defult_path.split('/')[-1]
        



        import json
        train_path = os.path.expanduser(defult_path)
        with open(train_path,'r') as f:
            data = json.load(f)
        inputs = []
        outputs = []
        for example in data['examples']:
            inputs.append(example['input'])
            outputs.append(example['target'])
        utils.seed_everything(seed)
        
        inputs, outputs = utils.shuffle_list(inputs, outputs)
        example_num = 5
            
        # get lora list by example and origin embedding
        example_inputs = inputs[0:example_num]
        example_outputs = outputs[0:example_num]
        example_embeddings_noanswer = embedding_model.encode([i for i,j in zip(example_inputs,example_outputs)])
        example_embeddings = embedding_model.encode([i+j for i,j in zip(example_inputs,example_outputs)])

        question_inputs = inputs[example_num:]
        question_outputs = outputs[example_num:]
        question_embeddings_noanswer = embedding_model.encode(question_inputs)


        example_adapted_lora_list = []
        #top 20 same lora
        for example_embedding in example_embeddings:
            centroid = torch.tensor(example_embedding)
            lora_index = torch.argsort(torch.cosine_similarity(candidate_lora_embedding, centroid,dim=1))[:20]
            lora_list = [list(candidate_lora_path)[i] for i in lora_index]
            example_adapted_lora_list.append(lora_list)

        lora_lists_index = []
        for question_embedding_noanswer in question_embeddings_noanswer:
            centroid = torch.tensor(question_embedding_noanswer)
            example_embeddings_noanswer = torch.tensor(example_embeddings_noanswer)
            example_index = torch.argsort(torch.cosine_similarity(example_embeddings_noanswer, centroid,dim=1))[0]
            lora_lists_index.append(example_index)

        quesiton_input_chunk = []
        #rechunk question_inputs
        for i in range(0, len(example_inputs)):
            chunk = []
            for index,input,output in zip(lora_lists_index,question_inputs,question_outputs):
                if index == i:
                    chunk.append((input,output))
            quesiton_input_chunk.append(chunk)

        all_currect = 0
        all_total = 0
        for lora_list,io in zip(example_adapted_lora_list,quesiton_input_chunk):
            if io == []:
                continue
            model, tokenizer, cache, base_model = algorithm.load_base_model_and_lora_modules(lora_list, args.model_name_or_path)
            step = 40
            prefs = {}
            #step 1
            module_weights,model,tokenizer = algorithm.lorahub_learning(model=model,
                                                                        cache=cache,
                                                                        tokenizer=tokenizer,
                                                                        lora_module_list=lora_list,
                                                                        example_inputs=example_inputs,
                                                                        example_outputs=example_outputs,
                                                                        max_inference_step=step,
                                                                        start_weights=0.0,
                                                                        batch_size=5,
                                                                        mode='origin',
                                                                        )

            
            # model_weight_stat[i] = module_weights
            valid_inputs = [x[0] for x in io]
            valid_outputs = [x[1] for x in io]
            example_predictions, perf = algorithm.lorahub_inference(example_inputs=valid_inputs,
                                                            model_or_name_path=model,
                                                            tokenizer_or_tokenizer_path=tokenizer,
                                                            batch_size=10,
                                                            example_outputs=valid_outputs,
                                                            detail=True)
            pref,correct,total = perf
            all_currect += correct
            all_total += total
        

        pref = all_currect/all_total * 100
        task_accs[task_name] = pref
        
        print("task accuracy:", perf)

    utils.save_results(f'~/resourses/results/seed{seed}_super_special_example_center_same_init00_step_{step}_cosin.result', task_accs)

