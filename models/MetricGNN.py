import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class Gconv(nn.Module):
    def __init__(self, input_dim, output_dim, args, use_bias=True):

        super(Gconv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.args = args

        self.weight = nn.Parameter(torch.Tensor(args.batch_size, input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(args.batch_size, args.data_len, output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_feature, adjacency):
        #print(input_feature.shape, self.weight.shape)
        support = torch.bmm(input_feature, self.weight)  # XW (N,D');X (N,D);W (D,D')
        output = torch.bmm(adjacency.squeeze(3), support)  # (N,D')
        #print(input_feature.shape,output.shape)
        if self.use_bias:
            output += self.bias
        return output



class Wcompute(nn.Module):
    def __init__(self, input_features, nf, activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=False):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf*ratio[2])
        self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf*ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation

    def forward(self, x):
        W1 = x.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2) #size: bs x N x N x num_features
        W_new = torch.abs(W1 - W2) #size: bs x N x N x num_features
        W_new = torch.transpose(W_new, 1, 3) #size: bs x num_features x N x N

        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1



        if self.activation == 'softmax':
            W_new = F.softmax(W_new,dim=2)

        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)

        elif self.activation == 'none':
            W_new = W_new
        else:
            raise (NotImplementedError)


        return W_new


class GNN_ml(nn.Module):
    def __init__(self, args, input_features, nf):
        super(GNN_ml, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.num_layers = 2


        for i in range(self.num_layers):

            module_w = Wcompute(self.input_features, nf, activation='sigmoid', ratio=[2, 2, 1, 1])
            module_l = Gconv(self.input_features, self.input_features,self.args)

            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features, nf,  activation='sigmoid', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features, 1,self.args)

    def forward(self, x):


        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x)

            x = F.leaky_relu(self._modules['layer_l{}'.format(i)](x, Wi))

            #print(i,x.shape)

        Wl=self.w_comp_last(x)
        out = self.layer_last(x, Wl)

        return out,Wl


class MetricGNN(nn.Module):
    def __init__(self, args):
        super(MetricGNN,self).__init__()


        self.emb_size = args.emb_size
        self.args = args
        num_inputs = self.emb_size
        self.gnn_obj = GNN_ml(args, num_inputs, nf=96)

    def forward(self, inputs):
        # Creating WW matrix


        nodes = [node.unsqueeze(1) for node in inputs]
        nodes = torch.cat(nodes, 1)
        logits, Wlogits = self.gnn_obj(nodes)

        outputs = torch.sigmoid(logits)

        return outputs, logits, Wlogits


# test

import argparse

parser = argparse.ArgumentParser(description = "GCN with metric learning for view graph")
parser.add_argument('--train',)
parser.add_argument('--exp_name',type=str, default ='test')

#network
parser.add_argument('--emb_size', type = int, default= 768, help = 'Embedding size')


#dataset
parser.add_argument('--dataset_root', type= str, default='/Volumes/Data2/GL3D/data', help='absolute dir of data')
parser.add_argument('--traindata_list', type = str, default='./data/train', help='graph data list in txt')
parser.add_argument('--testdata_list', type = str, default='./data/test')
parser.add_argument('--data_len', type = int, default= 2)
# training
parser.add_argument('--batch_size', type = int, default= 1)


args = parser.parse_args()


if __name__ == '__main__':
    nodes = []
    for i in range(2):
        nodes.append(torch.rand((1,768)))

    m = MetricGNN(args)
    out = m(nodes)

    print(out[0].shape,out[1].shape,out[2].shape)