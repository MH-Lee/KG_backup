import json
import argparse
import os.path as osp
from helper import *
from dataloader import *
from model.models import *
from torch.utils.data import DataLoader
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Runner(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if args.device != 'cpu' else 'cpu')
        self.load_data()
        if self.args.model_name == 'conve':
            print('model_name: ', 'ConvE')
            self.model = ConvE(self.args, self.args.num_ent, self.args.num_rel).to(self.device)
        else:
            print('model_name: ', 'CTE')
            self.model = CTE(self.args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2)
    
    def load_data(self):
        """
        Parameters
		----------
		self.args.dataset:         Takes in the name of the dataset (FB15k-237)
		
		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.embed_dim:         Embedding dimension used
  
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['valid']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits
        """
        ent_set, rel_set = OrderedSet(), OrderedSet()
        self.data, sr2o = ddict(list), ddict(set)
        with open(osp.join(self.args.data_dir, self.args.dataset, 'entities.json'), 'r') as en_dict:
            self.ent2id = json.load(en_dict)
        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}

        with open(osp.join(self.args.data_dir, self.args.dataset, 'relations.json'), 'r') as rel_dict:
            self.rel2id = json.load(rel_dict)
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for rel, idx in self.rel2id.items()})
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
        
        
        self.args.num_ent = len(self.ent2id)
        self.args.num_rel = len(self.rel2id)
        rel_number = len(self.rel2id) // 2
        
        print('dataset: ', self.args.dataset)
        print('number of entities: ', self.args.num_ent)
        print('number of relations: ', self.args.num_rel)
        
        train_edge_num  = 0
        unique_train = set()
        for split in ['train', 'valid', 'test']:
            print(osp.join(self.args.data_dir, self.args.dataset, f'{split}.txt'))
            with open(osp.join(self.args.data_dir, self.args.dataset, f'{split}.txt'), 'r') as f:
                triple_list = [x.split('\t') for x in f.read().split('\n')[:-1]]
            for sub, rel, obj in triple_list:
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train': 
                    train_edge_num += 1
                    unique_train.add(sub)
                    unique_train.add(obj)
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + rel_number)].add(sub)
                else:
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + rel_number)].add(sub)
            if split == 'train':
                self.sr2o = {k: list(v) for k, v in sr2o.items()}
        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.data = dict(self.data)
       
        self.triples  = ddict(list)
        for (sub, rel, obj) in self.data['train']:
            self.triples['train'].append({'triple':(sub, rel, obj),  'label': self.sr2o[(sub, rel)]})
            self.triples['train'].append({'triple':(obj, rel + rel_number, sub),  'label': self.sr2o[(obj, rel + rel_number)]})
        
        for split in ['valid', 'test']:
            for (sub, rel, obj) in self.data[split]:
                self.triples['{}_{}'.format(split, 'tail')].append({'triple':(sub, rel, obj),  'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple':(obj, rel + rel_number, sub),  'label': self.sr2o_all[(obj, rel + rel_number)]})

        self.triples = dict(self.triples)
        
        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
				dataset_class(self.triples[split], self.args),
				batch_size=batch_size,
				shuffle=shuffle,
				num_workers=max(0, self.args.num_workers),
				collate_fn=dataset_class.collate_fn
			)
        self.data_iter = {
			'train': get_data_loader(TrainDataset, 'train', self.args.batch_size),
			'valid_head': get_data_loader(TestDataset, 'valid_head', self.args.batch_size, False),
			'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.args.batch_size, False),
			'test_head': get_data_loader(TestDataset, 'test_head', self.args.batch_size, False),
			'test_tail': get_data_loader(TestDataset, 'test_tail', self.args.batch_size, False),
		}

    
    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU
        
        Parameters
        ----------
        batch: 		the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split
        
        Returns
        -------
        Head, Relation, Tail, labels        
        """
        if split == 'train':
            triple, label, = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
         
    def evalate(self, split, epoch):
        left_results = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results = get_combined_results(left_results, right_results)
        return results
    
    def predict(self, split='valid', mode='tail_batch'):
        self.model.eval()
        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
            
            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred = self.model.forward(sub, rel)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] 	= target_pred
                
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0/ranks).item() + results.get('mrr', 0.0)
                for k in [1, 3, 5, 10]:
                    results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= (k)]) + results.get('hits@{}'.format(k), 0.0)
        return results
    
    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict'	: self.model.state_dict(),
            'best_val'	: self.best_val,
            'best_epoch'	: self.best_epoch,
            'optimizer'	: self.optimizer.state_dict(),
            'args'		: vars(self.args)
        }
        file_name = "{}_{}_{}_{}.pt".format(self.args.model_name, str(self.args.lr), str(self.best_val['mrr']), str(self.best_epoch))
        torch.save(state, osp.join(save_path, file_name))
    
    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        file_name = "{}_{}_{}_{}.pt".format(self.args.model_name, str(self.args.lr), str(self.best_val['mrr']), str(self.best_epoch))
        load_path = osp.join(load_path, file_name)
        state			= torch.load(load_path)
        state_dict		= state['state_dict']
        self.best_val		= state['best_val']
        self.best_val_mrr	= self.best_val['mrr'] 

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def run_epoch(self, epoch, val_mrr=0):
        self.model
        losses = []
        train_iter = iter(self.data_iter['train'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.max_epochs, eta_min=0.001)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, obj, label = self.read_batch(batch, 'train')
            
            pred = self.model.forward(sub, rel)
            loss = self.model.loss(pred, label)
            
            loss.backward()
            self.optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            
            if step % 100 == 0:
                print('Epoch: [{}|{}]: Train Loss {:.5}, Val MRR: {:.5}'.format(epoch, step, np.mean(losses), self.best_val_mrr))
        loss = np.mean(losses)
        print("Epoch: {} | Train Loss: {:.5}".format(epoch, loss))
        return loss
    
    def fit(self):
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join('./torch_saved', self.args.model_name)
        print("save_path: ", save_path)
        self.kill_cnt = 0
        
        for epoch in range(self.args.max_epochs):
            train_loss = self.run_epoch(epoch, val_mrr)
            val_results = self.evalate('valid', epoch)
            print(val_results)
            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt > self.args.patience:
                    break
                
            print('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))
        
        print('Loading best model, Evaluating on Test data')
        self.load_model(save_path)
        test_results = self.evalate('test', self.best_epoch)
        print(test_results)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments')
    parser.add_argument('--model_name', default='CTE', type=str, help='Model name to use default: CTE')
    parser.add_argument('--dataset', default='FB15K237', type=str, help='Dataset to use, default: FB15k-237')
    parser.add_argument("--data_dir", default='../dataset/', type=str, required=False, help="The input data dir.")
    parser.add_argument('--patience', default=10, type=int, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of processes to construct batches')
    parser.add_argument('--seed', type=int, default=5959, help='seed for random number generator')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cuda:1', 'cpu'], help='select cpu or gpu')
    parser.add_argument('--lbl_smooth', type=float, default=0.0,	help='Label Smoothing')
    parser.add_argument('--neg_num', default=100, type=int, help='Number of negative samples')
    
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--embed_dim', dest='embed_dim', default=200, type=int, help='Embedding dimension to give as input to score function')
    parser.add_argument('--lr', default=0.003, type=float, help='L2 Regularization for oprimizer')
    parser.add_argument('--l2', default=0.0, type=float, help='L2 Regularization for oprimizer')
    parser.add_argument('--bias', action='store_false', default=True, help='Whether to use bias in the model')
    parser.add_argument('--input_drop', default=0.2, type=float, help='Dropout to use in Input')
    parser.add_argument('--hidden_drop', default=0.3, type=float, help='ConvE: Hidden dropout')
    # parser.add_argument('--hidden_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('--feat_drop', default=0.2, type=float, help='Dropout in FC layer')
    parser.add_argument('--lr_decay', default=0.995, type=float, help='Learning rate Decay factor')
    parser.add_argument('--hidden_size', default=9728, type=int, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')
    parser.add_argument('--k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('--k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', default=200, type=int, help='ConvE: Number of filters')
    
    # attention 
    parser.add_argument('--emb_dim1', default=256, type=int)
    parser.add_argument('--emb_dim2', default=256, type=int)
    parser.add_argument('--nheads', default=1, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--nblocks', default=1, type=int)
    parser.add_argument('--attn_type', default='dual')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    model = Runner(args)
    model.fit()