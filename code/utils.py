import random
import numpy as np
import torch
def save_results(path, task_accs):
    # calculate avarage accuracy
    task_accs['average'] = sum(task_accs.values())/len(task_accs)
    with open(path,'w') as f:
        for i in task_accs:
            print(f'{i}:{task_accs[i]}')
            f.write(f'{i}:{task_accs[i]}\n')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def shuffle_list(*ls):
    l =list(zip(*ls))
    random.shuffle(l)
    return zip(*l)


def main():
    pass
if __name__ == '__main__':
    main()