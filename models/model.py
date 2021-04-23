import torch
import torch.nn as nn
import torch.nn.functional as F

from .Embedding import Custom_CNN
from .MetricGNN import MetricGNN


class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        if self.args.emb_method == 'CNN':
            self.emb = Custom_CNN(args)


        self.metric_nn = MetricGNN(args)

    def forward(self,input):

        if self.args.emb_method == 'CNN':
            batch_emb = [self.emb(img) for img in input]

        out_metric, out_logits, W_logits = self.metric_nn(batch_emb)

        return out_metric, out_logits,W_logits



def create_models(args):

    return GNN(args)