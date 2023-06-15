from torch.utils.data import Dataset
import numpy as np
import random
import torch


class TrainDataset(Dataset):
	def __init__(self, triples, args):
		self.triples = triples
		self.args = args
		self.lbl_smooth = self.args.lbl_smooth
		self.num_ent = self.args.num_ent
		self.entities = np.arange(self.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele = self.triples[idx]
		triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
		trp_label = self.get_label(label)
  
		if self.lbl_smooth != 0.0:
			trp_label = (1.0 - self.lbl_smooth) * trp_label + (1.0/ self.args.num_ent)

		return triple, trp_label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] for _ in data], dim=0)
		trp_label	= torch.stack([_[1] for _ in data], dim=0)
		return triple, trp_label
	
	def get_label(self, label):
		y = np.zeros([self.num_ent], dtype=np.float32)
		for e2 in label: 
			y[e2] = 1.0
		return torch.FloatTensor(y)


class TestDataset(Dataset):
	def __init__(self, triples,  args):
		self.triples = triples
		self.args = args
		self.num_ent = self.args.num_ent
  
	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele	= self.triples[idx]
		triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
		label = self.get_label(label)
		return triple, label

	@staticmethod
	def collate_fn(data):
		triple	= torch.stack([_[0] for _ in data], dim=0)
		label	= torch.stack([_[1] for _ in data], dim=0)
		return triple, label
	
	def get_label(self, label):
		y = np.zeros([self.num_ent], dtype=np.float32)
		for e2 in label: 
			y[e2] = 1.0
		return torch.tensor(y, dtype=torch.float32)