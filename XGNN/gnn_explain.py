import networkx as nx
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from pytorch_util import weights_init
from gcn import GCN
import torch.nn.functional as F
import copy
from policy_nn import PolicyNN
import gnns
import torch.optim as optim
from utils import progress_bar
import matplotlib.pyplot as plt
import classificationNetwork as cN
import random

class gnn_explain():
    def __init__(self, max_node, max_step, target_class, max_iters): 
        print('Start training pipeline')
        self.graph= nx.Graph() #graph object for the graph that is to build
   #$     self.mol = Chem.RWMol()  ### keep a mol obj, to check if the graph is valid 
        self.max_node = max_node #maximum number of nodes allowed in final graph
        self.max_step = max_step #maximum number of stepps allowed per iteration for adding nodes/edges
        self.max_iters = max_iters #total number of iterations for the training process
        self.num_class = 2 #number of classes the GNN model can predict
        self.node_type = 3#7 #number of different node types allowed in the graph
        self.learning_rate = 0.01 #for policy network
        self.roll_out_alpha = 2 #hyperparameter for balancing step rewards and future rewards from rollouts
        self.roll_out_penalty = -0.1 #penality if graph becomes invalid
        self.policyNets= PolicyNN(self.node_type, self.node_type) 
        self.gnnNets = cN.GCN(3,2,25) #gnns.DisNets() #pretrained GNN model
        self.reward_stepwise= 0.1 #step reward given for each successful action of adding an edge
        self.target_class = target_class #class teh GNN model should aim for in the final graph prediction
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.policyNets.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        
        self.dict = {0:"r",1:"g",2:"b"}
        self.convertNxToData = None
        self.checkpoint = "./checkpoint/ckpt.pth"
        self.starting_node = None
        
        #self.dict = {0:'C', 1:'N', 2:'O', 3:'F', 4:'I', 5:'Cl', 6:'Br'} #dictionary containing the different node types
        #self.color= {0:'g', 1:'r', 2:'b', 3:'c', 4:'m', 5:'w', 6:'y'}
        #self.max_poss_degree = {0: 4, 1: 5, 2: 2, 3: 1, 4: 7, 5:7, 6: 5}
        

    def train(self):
        ####given the well-trained model
        ### Load the model 
        checkpoint = torch.load(self.checkpoint)
        self.gnnNets.load_state_dict(checkpoint['net'])
        
        for i in range(self.max_iters):
            self.graph_reset() # graph is reset to the empty state
            for j in range(self.max_step):
                self.optimizer.zero_grad()#gradients zeroed out for backpropagation
                reward_pred = 0 #init rewards
                reward_step = 0
                n = self.graph.number_of_nodes() #current number of nodes
                if(n>self.max_node): # check if maximum nodes exceeded
                    break
                self.graph_old = copy.deepcopy(self.graph) #copy current graph
                ###get the embeddings
                X, A = self.read_from_graph(self.graph) #extracts node features and adjacency matrix 
           #     print('Current have', n, 'node')
                X = torch.from_numpy(X)
                A = torch.from_numpy(A)
                ### Feed to the policy nets for actions
             #   forward(self, node_feat, n2n_sp, node_num):
                 
                #use policy net to predict start node, tail node, logits is raw ouput of network
                start_action, start_logits_ori, tail_action, tail_logits_ori  = self.policyNets(X.float(), A.float(), n+self.node_type)

                #flag is used to track whether adding operation is success/valid.


                if(tail_action>=n): ####we need add node, then add edge
                    if(n==self.max_node):
                        flag = False #success flag set to false
                    else:
                        self.add_node(self.graph, n, tail_action.item()-n) #attemts to add node with the predicted type
                        #print(tail_action.item()-n)
                        flag = self.add_edge(self.graph, start_action.item(), n) #adds node between predicted start end end nodes
                else:
                    flag= self.add_edge(self.graph, start_action.item(), tail_action.item()) #adding edge between start and tail node
                
                if flag == True:
                    validity = self.check_validity(self.graph) #check whether grpah abides to rules
                
                
                if  flag == True: #### add edge  successfully
                    if validity == True:                        
                        reward_step = self.reward_stepwise #positive reward for successfull action
                        X_new, A_new = self.read_from_graph_raw(self.graph) #updated feature and adjacency matrix
                        X_new = torch.from_numpy(X_new)
                        A_new = torch.from_numpy(A_new)
                        
                        
                        data = self.convertNxToData(self.graph)       #*** Convert graph to data object
                        logits, probs = self.gnnNets(data.x,data.edge_index) #***self.gnnNets(X_new.float(), A_new.float()) #predicting class probs on created graph
                        #### based on logits, define the reward
                        prediction = logits.argmax(dim=1)#***_, prediction = torch.max(logits, 0)
                        if self.target_class == prediction:
                            reward_pred = probs.squeeze()[prediction] - 0.5 #### positive reward
                            
                            #print(prediction)
                            #print(probs)#[prediction])
                        else:
                            reward_pred = probs.squeeze()[self.target_class] - 0.5  ###negative reward
                        
                        ### Then we need to roll out.
                        reward_rollout= []
                        for roll in range(10):
                            reward_cur = self.roll_out(self.graph, j)
                            reward_rollout.append(reward_cur)
                        reward_avg = torch.mean(torch.stack(reward_rollout)) #averaging roll out rewards
                            ###desgin loss
                        total_reward = reward_step+reward_pred+reward_avg*self.roll_out_alpha  ## need to tune the hyper-parameters here. 
                        
                        if total_reward < 0:
                            self.graph = copy.deepcopy(self.graph_old) ### rollback
                     #   total_reward= reward_step+reward_pred
                        loss = total_reward*(self.criterion(start_logits_ori[None,:], start_action.expand(1)) 
                                + self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))
                    else:
                        total_reward = -1  # graph is not valid 
                        self.graph = copy.deepcopy(self.graph_old)
                        loss = total_reward*(self.criterion(start_logits_ori[None,:], start_action.expand(1)) 
                                + self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))                        
                else:
                    # ### case adding edge not successful
                        ### do not evalute
                    #    print('Not adding successful')
                    reward_step = -1
                    total_reward= reward_step+reward_pred
#                   #print(start_logits_ori)
                    #print(tail_logits_ori)
                    loss = total_reward*(self.criterion(start_logits_ori[None,:], start_action.expand(1)) + 
                            self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))               
            #    total_reward= reward_step+reward_pred
           #     loss = total_reward*(self.criterion(stop_logits[None,:], stop_action.expand(1)) + self.criterion(start_logits_ori[None,:], start_action.expand(1)) + self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policyNets.parameters(), 100)
                self.optimizer.step()
        #self.graph_draw(self.graph)
        plt.show()
        X_new, A_new = self.read_from_graph_raw(self.graph)
        X_new = torch.from_numpy(X_new)
        A_new = torch.from_numpy(A_new) 
        
        data = self.convertNxToData(self.graph)
        logits, probs = self.gnnNets(data.x,data.edge_index)#****self.gnnNets(X_new.float(), A_new.float())
        prob = probs.squeeze()[self.target_class].item()
        
        print(probs)
        print(prob)
        return data


    def graph_draw(self, graph):
        attr = nx.get_node_attributes(graph, "label")
        labels = {}
        color = []#color = ''
        for n in attr:
            labels[n]= self.dict[attr[n]]
            color.append(self.dict[attr[n]])
           
     #   labels=dict((n,) for n in attr)
        nx.draw(graph,node_color=color)#,labels=labels)#), node_color=color) ########################################
        
        
    def check_validity(self, graph):
        """
        node_types = nx.get_node_attributes(graph,'label')
        for i in range(graph.number_of_nodes()):
            degree = graph.degree(i)
            max_allow = self.max_poss_degree[node_types[i]]
            if(degree> max_allow):
                return False"""
        return True
    
    def roll_out(self, graph, j):
        cur_graph = copy.deepcopy(graph)
        step = 0
        while(cur_graph.number_of_nodes()<=self.max_node and step<self.max_step-j):
          #  self.optimizer.zero_grad()
            graph_old = copy.deepcopy(cur_graph)
            step = step + 1
            X, A = self.read_from_graph(cur_graph)
            n = cur_graph.number_of_nodes()
            X = torch.from_numpy(X)
            A = torch.from_numpy(A)
            start_action, start_logits_ori, tail_action, tail_logits_ori  = self.policyNets(X.float(), A.float(), n+self.node_type)
            if(tail_action>=n): ####we need add node, then add edge
                if(n==self.max_node):
                    flag = False
                else:
                    self.add_node(cur_graph, n, tail_action.item()-n)
                    flag = self.add_edge(cur_graph, start_action.item(), n)
            else:
                flag= self.add_edge(cur_graph, start_action.item(), tail_action.item())
                    
            ## if the graph is not valid in rollout, two possible solutions
            ## 1. return a negative reward as overall reward for this rollout  --- what we do here. 
            ## 2. compute the loss but do not update model parameters here--- update with the step loss togehter. 
            if flag == True:
                validity = self.check_validity(cur_graph)
                if validity == False:
                    return torch.tensor(self.roll_out_penalty)
                    #cur_graph = copy.deepcopy(graph_old)
                    # total_reward = -1
                    # loss = total_reward*(self.criterion(start_logits_ori[None,:], start_action.expand(1)) 
                    #             + self.criterion(tail_logits_ori[None,:], tail_action.expand(1))) 
                    # self.optimizer.zero_grad()
                    # loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.policyNets.parameters(), 100)
                    ## self.optimizer.step()                       
            else:  ### case 1: add edges but already exists, case2: keep add node when reach max_node
                return torch.tensor(self.roll_out_penalty)
                # reward_step = -1
                # loss = reward_step*(self.criterion(start_logits_ori[None,:], start_action.expand(1)) + 
                #         self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))    
                # self.optimizer.zero_grad()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.policyNets.parameters(), 100)
                ## self.optimizer.step()     
                
        ###Then we evaluate the final graph
        X_new, A_new = self.read_from_graph_raw(cur_graph)
        X_new = torch.from_numpy(X_new)
        A_new = torch.from_numpy(A_new)
        
        data = self.convertNxToData(self.graph)
        logits, probs = self.gnnNets(data.x,data.edge_index) #***#self.gnnNets(X_new.float(), A_new.float())  
        ### Todo
        reward = probs.squeeze()[self.target_class] - 0.5
        return reward
        

    def add_node(self, graph, idx, node_type):
        graph.add_node(idx, label=node_type)
        return 
    
    def add_edge(self, graph, start_id, tail_id):
        if graph.has_edge(start_id, tail_id) or start_id==tail_id:
            return False
        else:
            graph.add_edge(start_id, tail_id)
            return True
    
    def read_from_graph(self, graph): ## read graph with added  candidates nodes
        n = graph.number_of_nodes()
     #   degrees = [val for (node, val) in self.graph.degree()]
        F = np.zeros((self.max_node+self.node_type, self.node_type))
        attr = nx.get_node_attributes(graph, "label")
        attr = list(attr.values())
        nb_clss  = self.node_type
        targets=np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
        F[:n,:]= one_hot_feature
    ### then get the onehot features for the candidates nodes
        F[n:n+self.node_type,:]= np.eye(self.node_type)      
        
        E = np.zeros([self.max_node+self.node_type, self.max_node+self.node_type])
        E[:n,:n] = np.asarray(nx.to_numpy_matrix(graph))
        E[:self.max_node+self.node_type,:self.max_node+self.node_type] += np.eye(self.max_node+self.node_type)
        return F, E


    def read_from_graph_raw(self, graph): ### do not add more nodes
        n = graph.number_of_nodes()
      #  F = np.zeros((self.max_node+self.node_type, 1))
        attr = nx.get_node_attributes(graph, "label")
        attr = list(attr.values())
        nb_clss  = self.node_type
        targets=np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
      #  F[:n+1,0] = 1   #### current graph nodes n + candidates set k=1 so n+1

        E = np.zeros([n, n])
        E[:n,:n] = np.asarray(nx.to_numpy_matrix(graph))
     #   E[:n,:n] += np.eye(n)

        return one_hot_feature, E

    def graph_reset(self):
        self.graph.clear()
        rand_label = random.randint(0,self.node_type-1)
        if self.starting_node:
            self.graph.add_node(0, label= self.starting_node)# rand_label)#0)#self.dict = {0:'C', 1:'N', 2:'O', 3:'F', 4:'I', 5:'Cl', 6:'Br'}
        else:
            self.graph.add_node(0, label= rand_label)
       # self.graph.add_edge(1, 3)
        self.step = 0
        return 
    
                       
              
       

                

                

                

