import argparse
import os
import sys
import algorithm
import json
import utils

input_seed = 133
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='google/flan-t5-large')
    return parser.parse_args()


args = parse_args()
for seed in range(9,12):
    print(f'seed:{seed}')

    with open('/root/dir/dataset/flan_v2_can_align.json','r') as f:
        lora_message = json.load(f)
    candidate_lora_path = lora_message.keys()
    #shuffle lora path
    candidate_lora_path = list(candidate_lora_path)
    import random
    utils.seed_everything(seed)
    random.shuffle(candidate_lora_path)
    lora_list = candidate_lora_path[:20]

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
        utils.seed_everything(input_seed)
        inputs, outputs = utils.shuffle_list(inputs, outputs)
        example_num = 5
            
        # get lora list by example and origin embedding
        example_inputs = inputs[0:example_num]
        example_outputs = outputs[0:example_num]
        #top 20 same lora
        lora_list = lora_list
        # print(lora_index)

        # lora_list = [os.path.join(os.path.expanduser('~/dir/scripts/new_lora_hub/lora_model'), i) for i in os.listdir(os.path.expanduser('~/dir/scripts/new_lora_hub/lora_model'))]
        accs = {}
        model, tokenizer, cache, base_model = algorithm.load_base_model_and_lora_modules(lora_list, args.model_name_or_path)
        times = {}
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
                                                                    mode='origin',
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
        
    utils.save_results(f'/root/dir/scripts/remote/iLoraComp/resourses/results/seed{seed}_lorahub.result', task_accs)

        
