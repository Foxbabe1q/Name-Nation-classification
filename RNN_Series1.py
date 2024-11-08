import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = 18
        self.rnn = nn.RNN(input_size, hidden_size, num_layers = num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        output = output[:, -1, :]
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return hidden

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = 18
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden, c):
        output, (hidden, c) = self.rnn(x, (hidden, c))
        output = output[:, -1, :]
        output = self.fc(output)
        return output, hidden, c

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return hidden, c0


class SimpleBILSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleBILSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = 18
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, x, hidden, c):
        output, (hidden, c) = self.rnn(x, (hidden, c))
        output = output[:, -1, :]
        output = self.fc(output)
        return output, hidden, c

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers*2, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
        return hidden, c0



class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = 18
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        output = output[:, -1, :]
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return hidden