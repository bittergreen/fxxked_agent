import torch
import torch.nn as nn
import torch.nn.functional as F


class Brain(nn.Module):
    """
    JUST FOR FUN
    """

    def __init__(self) -> None:
        super().__init__()


class SurvivalLSTM(Brain):

    def __init__(self, input_size=2, hidden_size=16, output_size=4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_state = None
        self.cell_state = None

    def forward(self, x):
        if self.hidden_state is None or self.cell_state is None:
            # initialize
            batch_size = x.size(0)  # get batch size
            h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)  # (num_layers, batch_size, hidden_size)
            c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)  # (num_layers, batch_size, hidden_size)
        else:
            h0 = self.hidden_state
            c0 = self.cell_state

        # forward
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # update hidden states
        self.hidden_state = h_n.detach()  # detach from the compute graph to avoid gradient flow
        self.cell_state = c_n.detach()

        out = self.fc(out[:, -1, :])
        out = F.softmax(out, dim=-1)
        return out

    def reset_hidden_state(self):
        self.hidden_state = None
        self.cell_state = None

