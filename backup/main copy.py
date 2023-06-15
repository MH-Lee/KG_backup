import json
import argparse
import os.path as osp
from helper import *
from dataloader import *
from model.models import *
from torch.utils.data import DataLoader


class Runner(object):
    def __init__(self, args):
        self.args = args
        if (self.args.device.find('cuda') == 0) and torch.cuda.is_available():
            self.device = torch.device(self.args.device)
            print('Using CUDA')
            print(self.device)
        else:
            self.device = torch.device('cpu')
        self.load_data()
        self.model = ConvE(self.args, self.args.num_ent, self.args.num_rel).to(self.device)
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
        self.args.num_rel = len(self.rel2id) // 2
        self.args.embed_dim = self.args.k_w * self.args.k_h if self.args.embed_dim is None else self.args.embed_dim
        
        print('dataset: ', self.args.dataset)
        print('number of entities: ', self.args.num_ent)
        print('number of relations: ', self.args.num_rel)
        
        self.data = ddict(list)
        sr2o = ddict(set)
        
        for split in ['train', 'valid', 'test']:
            for line in open(osp.join(self.args.data_dir, self.args.dataset, f'{split}.txt'), 'r'):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))
                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel+self.args.num_rel)].add(sub)
        
        self.data = dict(self.data)
        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.args.num_rel)].add(sub)
        
        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)
        
        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple': (sub, rel, -1), 'label': obj, 'sub_samp':1})
        
        for split in ['valid', 'test']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.args.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})
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
        
        # self.edge_index, self.edge_type = self.construct_adj()
        
    # def construct_adj(self):
    #     """
    #     Constructor of the runner class
        
    #     Parameters
    #     ----------
        
    #     Returns
    #     -------
    #     Constructs the adjacency matrux for GCN
    #     """
    #     edge_index, edge_type = [], []
        
    #     for sub, rel, obj in self.data['train']:
    #         edge_index.append((sub, obj))
    #         edge_type.append(rel)
            
    #     for sub, rel, obj in self.data['train']:
    #         edge_index.append((obj, sub))
    #         edge_type.append(rel + self.args.num_rel)
            
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device).t()
    #     edge_type = torch.tensor(edge_type, dtype=torch.long).to(self.device)
    #     return edge_index, edge_type
    
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
            triple, label = [ _.to(self.device) for _ in batch]
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
                    results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k]).item() + results.get('hits@{}'.format(k), 0.0)
        return results
    
    def run_epoch(self, epoch, val_mrr=0):
        self.model
        losses = []
        train_iter = iter(self.data_iter['train'])
        
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, obj, label = self.read_batch(batch, 'train')
            
            pred = self.model.forward(sub, rel)
            loss = self.model.loss(pred, label)
            
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
            if step % 100 == 0:
                print('Epoch: {}|{}]: Train Loss {:.5}, Val MRR: {:.5}'.format(epoch, step, np.mean(losses), self.best_val_mrr))
        loss = np.mean(losses)
        print("Epoch: {} | Train Loss: {:.5}".format(epoch, loss))
        return loss
    
    def fit(self):
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        self.kill_cnt = 0
        
        for epoch in range(self.args.max_epochs):
            train_loss = self.run_epoch(epoch, val_mrr)
            val_results = self.evalate('valid', epoch)
            
            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                kill_cnt = 0
                
            else:
                kill_cnt += 1
                if kill_cnt > self.args.patience:
                    break
                
            print('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))
        
        print('Loading best model, Evaluating on Test data')
        test_results = self.evalate('test', self.best_epoch)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments')
    parser.add_argument('--dataset', default='FB15K237', type=str, help='Dataset to use, default: FB15k-237')
    parser.add_argument("--data_dir", default='../dataset/', type=str, required=False, help="The input data dir.")
    parser.add_argument('--patience', default=10, type=int, help='early stopping patience')
    parser.add_argument('--num_workers',	type=int, default=0, help='Number of processes to construct batches')
    parser.add_argument('--seed', type=int, default=5959, help='seed for random number generator')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'], help='select cpu or gpu')
    parser.add_argument('--lbl_smooth', type=float,     default=0.1,	help='Label Smoothing')
    
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=500, help='Number of epochs')
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
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    model = Runner(args)
    model.fit()