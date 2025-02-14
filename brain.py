import torch
import torch.nn as nn
import torch.nn.functional as F


class Brain(nn.Module):

    def __init__(self):
        super().__init__()


class SurvivalLSTM(Brain):

    def __init__(self, input_size=2, hidden_size=32, output_size=4):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        out = F.softmax(out, dim=-1)
        return out

