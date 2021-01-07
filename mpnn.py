import torch
import torch.nn as nn
import numpy as np
from rlcore.distributions import Categorical
import torch.nn.functional as F
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class GatedMPNN(nn.Module):
    def __init__(self, action_space, num_agents, num_entities, input_size=16, hidden_dim=128, embed_dim=None,
                 pos_index=2, norm_in=False, nonlin=nn.ReLU, n_heads=1, mask_dist=None, entity_mp=False):
        super().__init__()

        self.h_dim = hidden_dim
        self.nonlin = nonlin
        self.num_agents = num_agents  # 智能体数量
        self.num_entities = num_entities  # entities 数量
        self.K = 3  # 信息传递数
        self.embed_dim = self.h_dim if embed_dim is None else embed_dim
        self.n_heads = n_heads
        self.mask_dist = mask_dist
        self.input_size = input_size
        self.entity_mp = entity_mp

        self.pos_index = pos_index

        self.encoder = nn.Sequential(nn.Linear(self.input_size, self.h_dim),
                                     self.nonlin(inplace=True))

        self.messages = MultiHeadAttention(n_heads=self.n_heads, input_dim=self.h_dim, embed_dim=self.embed_dim)

        self.update = nn.Sequential(nn.Linear(self.h_dim + self.embed_dim, self.h_dim),
                                    self.nonlin())

        self.value_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                        self.nonlin(inplace=True),
                                        nn.Linear(self.h_dim, 1))

        self.policy_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                         self.nonlin(inplace=True))

        # 加了门
        self.gate = nn.Sequential(nn.Linear(self.h_dim + self.embed_dim, 1),
                                  nn.Sigmoid())

        if self.entity_mp:
            self.entity_encoder = nn.Sequential(nn.Linear(2, self.h_dim),
                                                self.nonlin(inplace=True))

            self.entity_messages = MultiHeadAttention(n_heads=1, input_dim=self.h_dim, embed_dim=self.embed_dim)

            self.entity_update = nn.Sequential(nn.Linear(self.h_dim + self.embed_dim, self.h_dim),
                                               self.nonlin(inplace=True))

        num_actions = action_space.n
        self.dist = Categorical(self.h_dim, num_actions)

        self.is_recurrent = False

        # 测试
        # print('input_size', input_size)    #simple spread: 4

        if norm_in:
            self.in_fn = nn.BatchNorm1d(self.input_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.apply(weights_init)

        self.attn_mat = np.ones((num_agents, num_agents))

        self.dropout_mask = None

    def calculate_mask(self, inp):
        # inp 的 shape: batch_size x self.input_size，其中batch_size = 进程数*智能体数
        pos = inp[:, self.pos_index:self.pos_index + 2]
        bsz = inp.size(0) // self.num_agents
        mask = torch.full(size=(bsz, self.num_agents, self.num_agents), fill_value=0, dtype=torch.uint8)

        if self.mask_dist is not None and self.mask_dist > 0:
            for i in range(1, self.num_agents):
                shifted = torch.roll(pos, -bsz * i, 0)
                dists = torch.norm(pos - shifted, dim=1)
                restrict = dists > self.mask_dist
                for x in range(self.num_agents):
                    mask[:, x, (x + i) % self.num_agents].copy_(restrict[bsz * x:bsz * (x + 1)])

        elif self.mask_dist is not None and self.mask_dist == -10:
            if self.dropout_mask is None or bsz != self.dropout_mask.shape[
                0] or np.random.random_sample() < 0.1:  # dropout
                temp = torch.rand(mask.size()) > 0.85
                temp.diagonal(dim1=1, dim2=2).fill_(0)
                self.dropout_mask = (temp + temp.transpose(1, 2)) != 0
            mask.copy_(self.dropout_mask)

        return mask

    def _fwd(self, inp):
        # inp 形状应该时 (batch_size,input_size)
        # inp - {iden, vel(2), pos(2), entities(...)}
        agent_inp = inp[:, :self.input_size]
        mask = self.calculate_mask(agent_inp)  # 形状 <batch_size/N,N,N>，0允许通信，1不允许

        # 测试
        # print('inp', inp.shape)              # 3 agent: (96, 10)            4 agent: (128, 12)
        # print('agent_inp', agent_inp.shape)   # 3 agent: 有entity (96, 4), 无entity (96, 10)
        # 4 agent: 有 entity (128, 4)，无 entity (128, 12)

        h = self.encoder(agent_inp)  # should be (batch_size,self.h_dim)
        if self.entity_mp:
            landmark_inp = inp[:, self.input_size:]  # x,y pos of landmarks wrt agents
            # 应该是 (batch_size,self.num_entities,self.h_dim)
            he = self.entity_encoder(landmark_inp.contiguous().view(-1, 2)).view(-1, self.num_entities, self.h_dim)
            entity_message = self.entity_messages(h.unsqueeze(1), he).squeeze(1)  # 应该是 (batch_size,self.h_dim)
            h = self.entity_update(torch.cat((h, entity_message), 1))  # 应该是 (batch_size,self.h_dim)

        h = h.view(self.num_agents, -1, self.h_dim).transpose(0, 1)  # 应该是 (batch_size/N,N,self.h_dim)

        for k in range(self.K):
            m, attn = self.messages(h, mask=mask, return_attn=True)  # 应该是 <batch_size/N,N,self.embed_dim>
            hm_concat = torch.cat((h, m), 2)     # 应该是 <batch_size/N,N,self.h_dim + self.embed_dim>
            h = self.update(hm_concat)  # 应该是 <batch_size/N,N,self.h_dim>
            if k == self.K - 1:
                gate = self.gate(hm_concat)
                h = gate * h

        h = h.transpose(0, 1).contiguous().view(-1, self.h_dim)

        self.attn_mat = attn.squeeze().detach().cpu().numpy()
        return h  # 应该是 <batch_size, self.h_dim>

    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def _value(self, x):
        return self.value_head(x)

    def _policy(self, x):
        return self.policy_head(x)

    def act(self, inp, state, mask=None, deterministic=False):
        x = self._fwd(inp)
        value = self._value(x)
        dist = self.dist(self._policy(x))
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action).view(-1, 1)
        return value, action, action_log_probs, state

    def evaluate_actions(self, inp, state, mask, action):
        x = self._fwd(inp)
        value = self._value(x)
        dist = self.dist(self._policy(x))
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy, state

    def get_value(self, inp, state, mask):
        x = self._fwd(inp)
        value = self._value(x)
        return value


class MPNN(nn.Module):
    def __init__(self, action_space, num_agents, num_entities, input_size=16, hidden_dim=128, embed_dim=None,
                 pos_index=2, norm_in=False, nonlin=nn.ReLU, n_heads=1, mask_dist=None, entity_mp=False):
        super().__init__()

        self.h_dim = hidden_dim
        self.nonlin = nonlin
        self.num_agents = num_agents # 智能体数量
        self.num_entities = num_entities # entities 数量
        self.K = 3
        self.embed_dim = self.h_dim if embed_dim is None else embed_dim
        self.n_heads = n_heads
        self.mask_dist = mask_dist
        self.input_size = input_size
        self.entity_mp = entity_mp

        self.pos_index = pos_index

        self.encoder = nn.Sequential(nn.Linear(self.input_size,self.h_dim),
                                     self.nonlin(inplace=True))

        self.messages = MultiHeadAttention(n_heads=self.n_heads,input_dim=self.h_dim,embed_dim=self.embed_dim)

        self.update = nn.Sequential(nn.Linear(self.h_dim+self.embed_dim,self.h_dim),
                                    self.nonlin(inplace=True))

        self.value_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                        self.nonlin(inplace=True),
                                        nn.Linear(self.h_dim,1))

        self.policy_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                         self.nonlin(inplace=True))

        if self.entity_mp:
            self.entity_encoder = nn.Sequential(nn.Linear(2,self.h_dim),
                                                self.nonlin(inplace=True))
            
            self.entity_messages = MultiHeadAttention(n_heads=1,input_dim=self.h_dim,embed_dim=self.embed_dim)
            
            self.entity_update = nn.Sequential(nn.Linear(self.h_dim+self.embed_dim,self.h_dim),
                                               self.nonlin(inplace=True))
        
        num_actions = action_space.n
        self.dist = Categorical(self.h_dim,num_actions)

        self.is_recurrent = False

        # 测试
        # print('input_size', input_size)    #simple spread: 4


        if norm_in:
            self.in_fn = nn.BatchNorm1d(self.input_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.apply(weights_init)

        self.attn_mat = np.ones((num_agents, num_agents))

        self.dropout_mask = None

    def calculate_mask(self, inp):
        # inp 的 shape: batch_size x self.input_size，其中batch_size = 进程数*智能体数
        
        pos = inp[:, self.pos_index:self.pos_index+2]
        bsz = inp.size(0)//self.num_agents
        mask = torch.full(size=(bsz,self.num_agents,self.num_agents),fill_value=0,dtype=torch.uint8)
        
        if self.mask_dist is not None and self.mask_dist > 0: 
            for i in range(1,self.num_agents):
                shifted = torch.roll(pos,-bsz*i,0)
                dists = torch.norm(pos-shifted,dim=1)
                restrict = dists > self.mask_dist
                for x in range(self.num_agents):
                    mask[:,x,(x+i)%self.num_agents].copy_(restrict[bsz*x:bsz*(x+1)])
        
        elif self.mask_dist is not None and self.mask_dist == -10:
           if self.dropout_mask is None or bsz!=self.dropout_mask.shape[0] or np.random.random_sample() < 0.1:
               temp = torch.rand(mask.size()) > 0.85
               temp.diagonal(dim1=1,dim2=2).fill_(0)
               self.dropout_mask = (temp+temp.transpose(1,2))!=0
           mask.copy_(self.dropout_mask)

        return mask            


    def _fwd(self, inp):
        # inp 应该是 (batch_size,input_size)
        # inp - {iden, vel(2), pos(2), entities(...)}
        agent_inp = inp[:,:self.input_size]          
        mask = self.calculate_mask(agent_inp) # shape <batch_size/N,N,N> with 0 for comm allowed, 1 for restricted

        # 测试
        # print('inp', inp.shape)              # 3 agent: (96, 10)            4 agent: (128, 12)
        # print('agent_inp', agent_inp.shape)   # 3 agent: 有entity (96, 4), 无entity (96, 10)
        # 4 agent: 有 entity (128, 4)，无 entity (128, 12)


        h = self.encoder(agent_inp) # 应该是 (batch_size,self.h_dim)
        if self.entity_mp:
            landmark_inp = inp[:,self.input_size:] # x,y pos of landmarks wrt agents
            # 应该是 (batch_size,self.num_entities,self.h_dim)
            he = self.entity_encoder(landmark_inp.contiguous().view(-1,2)).view(-1,self.num_entities,self.h_dim) 
            entity_message = self.entity_messages(h.unsqueeze(1),he).squeeze(1) # 应该是 (batch_size,self.h_dim)
            h = self.entity_update(torch.cat((h,entity_message),1)) # 应该是 (batch_size,self.h_dim)

        h = h.view(self.num_agents,-1,self.h_dim).transpose(0,1) # 应该是 (batch_size/N,N,self.h_dim)
        
        for k in range(self.K):
            m, attn = self.messages(h, mask=mask, return_attn=True) # 应该是 <batch_size/N,N,self.embed_dim>
            h = self.update(torch.cat((h,m),2)) # 应该是 <batch_size/N,N,self.h_dim>
        h = h.transpose(0,1).contiguous().view(-1,self.h_dim)
        
        self.attn_mat = attn.squeeze().detach().cpu().numpy()
        return h # 应该是 <batch_size, self.h_dim> again

    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def _value(self, x):
        return self.value_head(x)

    def _policy(self, x):
        return self.policy_head(x)

    def act(self, inp, state, mask=None, deterministic=False):
        x = self._fwd(inp)
        value = self._value(x)
        dist = self.dist(self._policy(x))
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action).view(-1,1)
        return value,action,action_log_probs,state

    def evaluate_actions(self, inp, state, mask, action):
        x = self._fwd(inp)
        value = self._value(x)
        dist = self.dist(self._policy(x))
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value,action_log_probs,dist_entropy,state

    def get_value(self, inp, state, mask):
        x = self._fwd(inp)
        value = self._value(x)
        return value


class MultiHeadAttention(nn.Module):
    # 取自 https://github.com/wouterkool/attention-tsp/blob/master/graph_encoder.py
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None, return_attn=False):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h 应该是 (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"


        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)                      # (1, 96, 1, 128) (1, 32, 3, 128)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)                         # (1, 96, 3, 128) (1, 32, 3, 128)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # 测试
        # print('K shape', K.shape)
        # print('Q shape', Q.shape)
        # print('================')

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -math.inf

        attn = F.softmax(compatibility, dim=-1)            # (1, 32, 3, 3)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)                       # (1, 32, 3, 128)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)          # (32, 3, 128)
        
        if return_attn:
            return out, attn
        return out



