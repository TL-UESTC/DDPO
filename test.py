import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import os
import numpy as np

from datasets.dataset import AliExpressDataset,AliCCPDataset,IndustrialDataset
from models.ddpo_ccp import DDPO as DDPO_C
from models.ddpo_express import DDPO as DDPO_E
from models.ddpo_ours import DDPO as DDPO_I
from models.mddpo_ccp import MDDPO as MDDPO_C
from models.mddpo_express import MDDPO as MDDPO_E
from models.mddpo_ours import MDDPO as MDDPO_I

#For reproducibility
torch.manual_seed(999)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)

def get_dataset(name, path):
    if 'AliExpress' in name:
        return AliExpressDataset(path)
    elif 'ccp' in name:
        return AliCCPDataset(path)
    elif 'industrial' in name:
        return IndustrialDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    
    if name == 'ddpo' and 'AliExpress' in args.dataset_name:
        print("Model: DDPO")
        print("Dataset:",args.dataset_name)
        return DDPO_E(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(128, 64), tower_mlp_dims=(64, 32), task_num=task_num, dropout=0.2)
    elif name == 'ddpo' and 'cpp' in args.dataset_name:
        print("Model: DDPO")
        print("Dataset:",args.dataset_name)
        return DDPO_C(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(128, 64), tower_mlp_dims=(64, 32), task_num=task_num, dropout=0.2)
    elif name == 'ddpo' and 'industrial' in args.dataset_name:
        print("Model: DDPO")
        print("Dataset:",args.dataset_name)
        return DDPO_I(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(128, 64), tower_mlp_dims=(64, 32), task_num=task_num, dropout=0.2)
    elif name == 'mddpo' and 'AliExpress' in args.dataset_name:
        print("Model: MDDPO")
        print("Dataset:",args.dataset_name)
        return MDDPO_E(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(16,), tower_mlp_dims=(16, 8), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'mddpo' and 'cpp' in args.dataset_name:
        print("Model: MDDPO")
        print("Dataset:",args.dataset_name)
        return MDDPO_C(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(16,), tower_mlp_dims=(16, 8), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'mddpo' and 'industrial' in args.dataset_name:
        print("Model: MDDPO")
        print("Dataset:",args.dataset_name)
        return MDDPO_I(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(16,), tower_mlp_dims=(16, 8), task_num=task_num, expert_num=expert_num, dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)

class EarlyStopper(object):

    def __init__(self, num_trials, load_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.load_path = load_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.load_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(model, optimizer, data_loader, criterion, device,parao,parac,parar,paract,log_interval=100):
    pass

def test(model, data_loader, task_num, device):
    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    with torch.no_grad():
        for categorical_fields, numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
            y = model(categorical_fields, numerical_fields)
            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(y[i].tolist())
                loss_dict[i].extend(torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(), reduction='none').tolist())
    auc_results, loss_results = list(), list()
    for i in range(task_num):
        if i == 1:
            click = np.array(labels_dict[0])
            labels = np.array(labels_dict[1])
            predicts = np.array(predicts_dict[1])

            rlabels = labels[click == 1].tolist()
            rpredicts = predicts[click == 1].tolist()

            auc_results.append(roc_auc_score(rlabels, rpredicts))
        elif i==2:
            pred0 = np.array(predicts_dict[0])
            pred1 = np.array(predicts_dict[1])
            pres = pred0*pred1
            auc_results.append(roc_auc_score(labels_dict[2], pres.tolist()))
        else:
            auc_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
        
        loss_results.append(np.array(loss_dict[i]).mean())
    return auc_results, loss_results


def main(dataset_name,
         dataset_path,
         task_num,
         expert_num,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         device,
         save_dir,
         parao,
         parac,
         parar,
         paract):
    device = torch.device(device)
    train_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/train.csv')
    test_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/test.csv')
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    model = get_model(model_name, field_dims, numerical_num, task_num, expert_num, embed_dim).to(device)
    load_path=f'{save_dir}/{dataset_name}_{model_name}.pt'

    model.load_state_dict(torch.load(load_path,map_location=torch.device(args.device)))
    auc, loss = test(model, test_data_loader, task_num, device)
    f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding = 'utf-8')
    f.write('learning rate: {}\n'.format(learning_rate))
    for i in range(task_num):
        print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
        f.write('task {}, AUC {}, Log-loss {}\n'.format(i, auc[i], loss[i]))
    print('\n')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ccp', choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US','ccp','industrial'])
    parser.add_argument('--dataset_path', default='./data/')
    parser.add_argument('--model_name', default='ddpo', choices=['ddpo', 'mddpo'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--task_num', type=int, default=3)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=5) #Ali-CCP:5 Others:8
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--parao', type=float, default=1.0)
    parser.add_argument('--parac', type=float, default=1.0)
    parser.add_argument('--parar', type=float, default=1.0)
    parser.add_argument('--paract', type=float, default=1.0)
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.task_num,
         args.expert_num,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.parao,
         args.parac,
         args.parar,
         args.paract)
