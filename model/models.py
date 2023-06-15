import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


class ConvE(nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.args = args
        self.emb_e = nn.Embedding(num_entities, args.embed_dim, padding_idx=0, device=args.device)
        self.emb_rel = nn.Embedding(num_relations*2, args.embed_dim, padding_idx=0, device=args.device)
        self.inp_drop = nn.Dropout(args.input_drop)
        self.hidden_drop = nn.Dropout(args.hidden_drop)
        self.feature_map_drop = nn.Dropout2d(args.feat_drop)
        self.loss = nn.BCELoss()
        self.emb_dim1 = args.k_h
        self.emb_dim2 = args.embed_dim // self.emb_dim1
        
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.bias)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(args.embed_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        self.fc = nn.Linear(args.hidden_size, args.embed_dim)
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        print(num_entities, num_relations)
        
    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_e(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)
        
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred
        

class CTE(nn.Module):
    def __init__(self, args):
        super(CTE, self).__init__()
        self.args = args      
        self.emb_ent = None
        self.emb_rel = None
        self.loss = nn.BCELoss()
        self.init_emb()

    def init_emb(self):

        """Initialize the convolution layer and embeddings .

        Args:
            conv1: The convolution layer.
            fc: The full connection layer.
            bn0, bn1, bn2: The batch Normalization layer.
            inp_drop, hid_drop, feg_drop: The dropout layer.
            emb_ent: Entity embedding, shape:[num_ent, emb_dim].
            emb_rel: Relation_embedding, shape:[num_rel, emb_dim].
        """
        self.emb_dim1 = self.args.emb_dim1
        self.emb_dim2 = self.args.emb_dim2
        self.nheads = self.args.nheads
        self.inp_drop = self.args.input_drop
        self.attn_type = self.args.attn_type
        self.emb_ent = nn.Embedding(self.args.num_ent, self.emb_dim1, padding_idx=0)
        self.emb_rel = nn.Embedding(self.args.num_rel, self.emb_dim2, padding_idx=0)
        nn.init.xavier_normal_(self.emb_ent.weight.data)
        nn.init.xavier_normal_(self.emb_rel.weight.data)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.args.num_ent)))

        if self.emb_dim1 == self.emb_dim2:
            add_bias_kv = False
            kdim1 = None
            vdim1 = None
            kdim2 = None
            vdim2 = None
        else:
            add_bias_kv = True
            kdim1 = self.emb_dim2
            vdim1 = self.emb_dim2
            kdim2 = self.emb_dim1
            vdim2 = self.emb_dim1

        ### initial layer norm
        self.attn_layernorms_ent = nn.ModuleList()
        self.attn_layernorms_rel = nn.ModuleList()
        ### skip_connection layer norm
        self.sc_layernorms_ent = nn.ModuleList()
        self.sc_layernorms_rel = nn.ModuleList()
        ### attnetion layer
        self.attn_layers_ent = nn.ModuleList()
        self.attn_layers_rel = nn.ModuleList()
        ### ffn intermediate layers
        self.ffn_int_layers = nn.ModuleList()
        
        for layer in range(self.args.nblocks):
            layer_norm_ent = nn.LayerNorm(self.emb_dim1, eps=1e-8)
            self.attn_layernorms_ent.append(layer_norm_ent)
            attn_ent = nn.MultiheadAttention(embed_dim=self.emb_dim1, \
                                             num_heads=self.nheads, \
                                             dropout=self.inp_drop,\
                                             batch_first=True,\
                                             add_bias_kv=add_bias_kv,\
                                             kdim=kdim1, vdim=vdim1)
            self.attn_layers_ent.append(attn_ent)
            sc_layer_norm_ent = nn.LayerNorm(self.emb_dim1, eps=1e-8)
            self.sc_layernorms_ent.append(sc_layer_norm_ent)
            if self.attn_type == 'dual':
                layer_norm_rel = nn.LayerNorm(self.emb_dim2, eps=1e-8)
                self.attn_layernorms_rel.append(layer_norm_rel)
                attn_rel = nn.MultiheadAttention(embed_dim=self.emb_dim2, \
                                                num_heads=self.nheads, \
                                                dropout=self.inp_drop,\
                                                batch_first=True,\
                                                add_bias_kv=add_bias_kv,\
                                                kdim=kdim2, vdim=vdim2)
                self.attn_layers_rel.append(attn_rel)
                sc_layer_norm_rel = nn.LayerNorm(self.emb_dim2, eps=1e-8)
                self.sc_layernorms_rel.append(sc_layer_norm_rel)
                ffn_in_channels = self.emb_dim1 + self.emb_dim2    
            else:
                ffn_in_channels = self.emb_dim1
            ffn_intermediate = nn.Sequential(nn.Linear(ffn_in_channels, ffn_in_channels), nn.GELU())
            self.ffn_int_layers.append(ffn_intermediate)
            
        out_channels = self.emb_dim1
        self.hid_drop = nn.Dropout(self.args.hidden_drop)
        self.ffn_output = nn.Sequential(nn.Linear(ffn_in_channels, self.args.dim_feedforward),
                                        nn.GELU(),
                                        nn.Linear(self.args.dim_feedforward, out_channels))
        self.last_layernorms = nn.LayerNorm(out_channels, eps=1e-8)

    def forward(self, e1, rel):

        """Calculate the score of the triple embedding.

        This function calculate the score of the embedding.
        First, the entity and relation embeddings are reshaped
        and concatenated; the resulting matrix is then used as
        input to a convolutional layer; the resulting feature
        map tensor is vectorised and projected into a k-dimensional
        space.

        Args:
            head_emb: The embedding of head entity.
            relation_emb:The embedding of relation.

        Returns:
            score: Final score of the embedding.
        """
        head_emb = self.emb_ent(e1).view(-1, 1, self.emb_dim1)
        rel_emb = self.emb_rel(rel).view(-1, 1, self.emb_dim2)

        for layer in range(self.args.nblocks):
            if self.attn_type == 'entity-query':
                Q_ent = self.attn_layernorms_ent[layer](head_emb)
                attn_out, _ = self.attn_layers_ent[layer](Q_ent, rel_emb, rel_emb)
                head_emb = Q_ent + attn_out
                head_emb = self.sc_layernorms_ent[layer](head_emb)
                head_emb = self.ffn_int_layers[layer](head_emb)
            elif self.attn_type == 'dual':
                Q_ent = self.attn_layernorms_ent[layer](head_emb)
                Q_rel = self.attn_layernorms_rel[layer](rel_emb)
                ent_attn_out, _ = self.attn_layers_ent[layer](Q_ent, rel_emb, rel_emb)
                rel_attn_out, _ = self.attn_layers_rel[layer](Q_rel, head_emb, head_emb)
                ent_attn_res = self.sc_layernorms_ent[layer](Q_ent + ent_attn_out)
                rel_attn_res = self.sc_layernorms_rel[layer](Q_rel + rel_attn_out)                
                merged_out = torch.cat((ent_attn_res, rel_attn_res), dim=-1)
                if layer <= (self.args.nblocks - 1):
                    merged_out = self.ffn_int_layers[layer](merged_out)
                    head_emb, relation_emb = torch.split(merged_out, [self.emb_dim1, self.emb_dim2], dim=-1)
            else:
                raise NotImplementedError("select attention type (1) entity query (2) dual")
        if self.attn_type == 'entity-query':
            out = head_emb.squeeze()      
        elif self.attn_type == 'dual':
            out = merged_out.squeeze()
        x = self.ffn_output(out)
        x = self.last_layernorms(x)
        x = self.hid_drop(x)
        x = torch.mm(x, self.emb_ent.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred