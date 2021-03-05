import torch as t 
from torch.nn import GELU



class FeedForward(t.nn.Module):
    def __init__(self, input_size, inner_size, dropout):
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.inner_size = inner_size
        self.dropout = dropout
        self.linear1 = t.nn.Linear(input_size, inner_size)
        t.nn.init.xavier_normal_(self.linear1.weight)
        self.gelu = GELU()
        self.dropout1 = t.nn.Dropout(dropout, inplace=True)
        self.linear2 = t.nn.Linear(inner_size, input_size)
        t.nn.init.xavier_normal_(self.linear2.weight)
        self.dropout2 = t.nn.Dropout(dropout, inplace=True)
        self.rezero_alpha = t.nn.Parameter(t.Tensor([0]))

    def forward(self, net):
        residual = net
        net = self.linear1(net)
        net = self.gelu(net)
        net = self.dropout1(net)
        net = self.linear2(net)
        net = self.dropout2(net)
        net = self.rezero_alpha * net
        net += residual
        return net




# class FeedForward(t.nn.Module):
#     def __init__(self, input_size, inner_size, dropout):
#         super(FeedForward, self).__init__()
#         self.input_size = input_size
#         self.inner_size = inner_size
#         self.dropout = dropout

#         self.linear1 = t.nn.Conv1d(input_size, inner_size, kernel_size=3, stride=1, padding=1)
#         t.nn.init.kaiming_normal_(self.linear1.weight)
#         self.gelu = Gelu()
#         self.dropout = t.nn.Dropout(dropout, inplace=True)
#         self.linear2 = t.nn.Conv1d(inner_size, input_size, kernel_size=3, stride=1, padding=1)
#         t.nn.init.kaiming_normal_(self.linear2.weight)
#         self.dropout = t.nn.Dropout(dropout, inplace=True)
#         self.rezero_alpha = t.nn.Parameter(t.Tensor([0]))

#     def forward(self, net):
#         residual = net
#         net = self.linear1(net.transpose(1, 2))
#         net = self.gelu(net)
#         net = self.dropout(net)
#         net = self.linear2(net).transpose(1, 2)
#         net = self.dropout(net)
#         net = self.rezero_alpha * net
#         net += residual
#         return net