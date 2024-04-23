import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from pytorch_util import weights_init
from gcn import GCN
import torch.nn.functional as F

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    #print("===Normalizing adjacency matrix symmetrically===")
    adj = adj.numpy()
    N = adj.shape[0]
  #  adj = adj + np.eye(N)
    D = np.sum(adj, 0)
    D_hat = np.diag(np.power(D,-0.5))
 #   np.diag((D )**(-0.5))
    out = np.dot(D_hat, adj).dot(D_hat)
    out[np.isnan(out)]=0
    out = torch.from_numpy(out)
    return out, out.float()




class PolicyNN(nn.Module):
    def __init__(self,  input_dim, node_type_num, initial_dim=8, latent_dim=[16, 24, 32],  max_node = 12): 
        print('Initializing Policy Nets')
        super(PolicyNN, self).__init__() #init of base neural network class

        self.latent_dim = latent_dim #hidden layer dim
        self.input_dim  = input_dim #dimensionality of input feature for each node in the graph
        self.node_type_num =node_type_num #number of different node types in graph
        self.initial_dim = initial_dim #The initial dimensionality of the embedding after the first MLP layer.
  #      self.stop_mlp_hidden = 16
        self.start_mlp_hidden = 16 # The hidden layer size for the MLP predicting the starting action probability.
        self.tail_mlp_hidden = 24 #The hidden layer size for the MLP predicting the ending action probability.

        self.input_mlp = nn.Linear(self.input_dim, initial_dim) #creating the initial fully connected layer


        self.gcns = nn.ModuleList() #list for holding the layers
        self.layer_num = len(latent_dim) #number of gnn layers
        self.gcns.append(GCN(self.initial_dim, self.latent_dim[0])) #first layer is a fc layer
        for i in range(1, len(latent_dim)): #gnn layers
            self.gcns.append(GCN(self.latent_dim[i-1], self.latent_dim[i]))
        
        self.dense_dim = latent_dim[-1] #outgoing dim

        # self.stop_mlp1 = nn.Linear(self.dense_dim, self.stop_mlp_hidden)
        # self.stop_mlp_non_linear= nn.ReLU6()
        # self.stop_mlp2 = nn.Linear(self.stop_mlp_hidden, 2)

        #creating mlp layers for reinforcement learning
        self.start_mlp1= nn.Linear(self.dense_dim, self.start_mlp_hidden)
        self.start_mlp_non_linear = nn.ReLU6()
        self.start_mlp2= nn.Linear(self.start_mlp_hidden, 1) #output: probability for using node as a start node


        self.tail_mlp1= nn.Linear(2*self.dense_dim, self.tail_mlp_hidden)
        self.tail_mlp_non_linear = nn.ReLU6()
        self.tail_mlp2= nn.Linear(self.tail_mlp_hidden, 1)


        weights_init(self)


    def forward(self, node_feat, n2n_sp, node_num):
        
        un_A, A = normalize_adj(n2n_sp) #create symmetric normalized adjecency matrix
      #  A = n2n_sp
     #   print('adj has shape', A.size())
     #   print('node feature has shape', node_feat.size())
        cur_out = node_feat #feature matrix
        cur_A = A 

        cur_out = self.input_mlp(cur_out) # applies first mlp to node featues
  #      cur_out = self.input_non_linear(cur_out)


        for i in range(self.layer_num): #transforms embedding for every gnn layer
            cur_out = self.gcns[i](cur_A, cur_out)
        
        ### now we have the node embeddings

        ### get two different masks
        ob_len = node_num ##total current + candidates set
        ob_len_first = ob_len - self.node_type_num
        
        logits_mask = self.sequence_mask(ob_len, cur_A.size()[0])
        
        logits_mask_first = self.sequence_mask(ob_len_first, cur_A.size()[0])
   #     print('logits_mask_first has shape', logits_mask_first.size())
        graph_embedding = torch.mean(cur_out, 0) #calculates average of final node embeddings to get a graph embedding
      #  print('graph_embedding has shape', graph_embedding.size())


        ### action--- select the starting node, two layer mlps
        
        start_emb = self.start_mlp1(cur_out) 
        start_emb = self.start_mlp_non_linear(start_emb)
        start_logits = self.start_mlp2(start_emb) #final layer
        start_logits_ori = torch.squeeze(start_logits) #removes unnecessary dimensions from the logits tensor
    #    print('start_logits has shape', start_logits.size())
        start_logits_short = start_logits_ori[0:ob_len_first] #getting values for starting nodes
        
        start_probs = torch.nn.functional.softmax(start_logits_short,dim=0) #getting normalized probabilities
        
            
        start_prob_dist = torch.distributions.Categorical(start_probs) #creates probability distribution object
        try:
            start_action = start_prob_dist.sample() #samples start action (node)
        except:
            import pdb
            pdb.set_trace()
        

        mask = F.one_hot(start_action, num_classes=node_feat.size()[0]) #position of starting node is set to 1
        mask = mask.bool() #convert to boolean tensor
        emb_selected_node = torch.masked_select(cur_out,mask[:,None]) #uses mask to select embedding of chosen starting node from final node

        ### action--- select the tail node, two layer mlps
        emb_selected_node_copy = emb_selected_node.repeat(cur_out.size()[0],1) #Replicates the selected starting node embedding to have the same size as the current node embeddings (cur_out).
        cat_emb = torch.cat((cur_out, emb_selected_node_copy), 1) #Concatenates the current node embeddings with the replicated starting node embedding along the feature dimension.
        #This creates an augmented representation that includes information about both the graph structure and the chosen starting nod
        
        tail_emb = self.tail_mlp1(cat_emb)
        tail_emb = self.tail_mlp_non_linear(tail_emb)
        tail_logits= self.tail_mlp2(tail_emb)
        tail_logits_ori =torch.squeeze(tail_logits)


        logits_second_mask = logits_mask[0] & ~mask
        tail_logits_short =  tail_logits_ori[0:ob_len] 
        logits_second_mask_short = logits_second_mask[0:ob_len]
        
        
        tail_logits_null = torch.ones_like(tail_logits_short)*-1000000 
        tail_logits_short = torch.where(logits_second_mask_short==True, tail_logits_short, tail_logits_null)
        
        tail_probs = torch.nn.functional.softmax(tail_logits_short,dim=0)
        

        
        tail_prob_dist = torch.distributions.Categorical(tail_probs)
        #print(tail_prob_dist.probs)
        try:
            tail_action = tail_prob_dist.sample()
#            if tail_action >= start_action:
#                tail_action = tail_action +1
        except:
            import pdb
            pdb.set_trace()

        return start_action, start_logits_ori, tail_action, tail_logits_ori 


    def sequence_mask(self, lengths, maxlen, dtype=torch.bool):
        mask = ~(torch.ones((lengths, maxlen)).cumsum(dim=1).t() > lengths).t()
        mask.type(dtype)
        return mask