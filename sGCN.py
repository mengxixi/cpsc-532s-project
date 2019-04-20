import torch
from torch import nn

class sGCN(nn.Module):
    def __init__(self, input_size, l1_size, output_size):
        super(sGCN, self).__init__()
        self.input_size = input_size
        self.l1_size = l1_size
        self.output_size = output_size

        self.proj1 = nn.Linear(input_size, l1_size)
        self.proj2 = nn.Linear(l1_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, G, seq_lengths):        
        eyes = torch.zeros(G.shape[1], G.shape[1]).repeat(x.shape[0], 1, 1).cuda()
        for i, sl in enumerate(seq_lengths):
            eyes[i,:sl,:sl] = torch.eye(sl).cuda()

        Ghat = G + eyes
        D = torch.diag_embed(1/torch.sqrt(torch.sum(Ghat, dim=1)), dim1=1, dim2=2)
        D[D == float('inf')] = 0 # numerical stability
        C = torch.bmm(torch.bmm(D, Ghat), D)
        H1 = self.dropout(torch.relu(self.proj1(torch.bmm(C, x))))
        H2 = self.dropout(torch.relu(self.proj2(torch.bmm(C, H1))))

        return H2