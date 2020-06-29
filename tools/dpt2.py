import torch
from torch.nn.parallel import data_parallel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_gpu = torch.cuda.device_count()
print('Number of GPUs Available:', num_gpu)

def initHidden(batch_size, bidirectional, hidden_size, num_layers, device, num_gpu):
    '''
    This function is used to create a init vector for GRU/LSTMs
    '''
    if bidirectional:
        num_directions=2
    else:
        num_directions=1
    if num_gpu > 1:
        # The Dataparallel does split by default on dim=0 so we create like this to transpose
        # inside the model forward
        hidden = torch.zeros(batch_size, num_layers * num_directions, hidden_size, device=device)
        initial_cell = torch.zeros(batch_size, num_layers * num_directions, hidden_size, device=device)
        return hidden, initial_cell
    else:
        hidden = torch.zeros(num_layers * num_directions, batch_size, hidden_size, device=device)
        initial_cell = torch.zeros(num_layers * num_directions, batch_size, hidden_size, device=device)
        return hidden, initial_cell

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.GRU(300, 1024, 1, batch_first=True, bidirectional=True)
    def forward(self, x, hidden):
        if self.training:
            self.rnn.flatten_parameters()
        return self.rnn(x, hidden.permute(1,0,2).contiguous())  # N * T * hidden_dim


model = Model()
if num_gpu > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)

x = torch.rand(4, 52, 300, device='cuda')
hidden = initHidden(4, True, 1024, 1, device, num_gpu)

with torch.no_grad():
    model(x,hidden[0])
