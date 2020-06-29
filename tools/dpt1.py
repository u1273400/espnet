import torch
from torch.nn.parallel import data_parallel

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(300, 1024, 1, batch_first=True, bidirectional=True)
    def forward(self, x):
        self.rnn.flatten_parameters()
        return self.rnn(x)  # N * T * hidden_dim


model = Model().to('cuda')
x = torch.rand(4, 52, 300, device='cuda')

with torch.no_grad():
    data_parallel(model, x, range(2))
