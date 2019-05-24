import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderWithLSTM(nn.Module):
    def __init__(self, pedestrian_num, input_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderWithLSTM, self).__init__()
        # input_size = 2
        self.pedestrian_num = pedestrian_num
        self.input_size = input_size

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout)

    def forward(self, input_traces, pre_hiddens):
        """
        :param input_traces: size: (batch, pedestrian_num, 2), where 2 means (x, y)
        :param pre_hiddens: size: [self.pedestrian_num, (2, [self.n_layers, batch_size, self.hidden_size])]
        :return: encoder_traces: (batch, pedestrian_num, hidden_size)
                 next_hiddens, the same size as pre_hiddens
        """
        batch_size = input_traces.size()[0]
        next_hiddens = []
        encoder_traces = torch.zeros(batch_size, self.pedestrian_num, self.hidden_size)
        for i in range(self.pedestrian_num):
            input_trace = input_traces[:, i, :].unsqueeze(0)
            encoder_trace, next_hidden = self.lstm(input_trace, (pre_hiddens[i][0], pre_hiddens[i][1]))
            encoder_traces[:, i, :] = encoder_trace.squeeze(0)
            next_hiddens.append(next_hidden)

        return encoder_traces, next_hiddens

    def init_hidden(self, batch_size):
        return [[torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).cuda()
                 for _ in range(2)]
                for _ in range(self.pedestrian_num)]


class DecoderWithAttention(nn.Module):
    def __init__(self, pedestrian_num, target_size, hidden_size):
        super(DecoderWithAttention, self).__init__()

        self.pedestrian_num = pedestrian_num
        self.target_size = target_size
        self.hidden_size = hidden_size

    def forward(self, decoder_hiddens, cell_states, hidden_states):
        # input_traces: (batch, pedestrian_num, 2)
        # hidden_states: (batch, pedestrian_num, hidden_size)
        pass


class PredictionNet(nn.Module):
    def __init__(self, pedestrian_num, regression_size, hidden_size):
        super(PredictionNet, self).__init__()

        self.pedestrian_num = pedestrian_num
        self.regression_size = regression_size  # 2
        self.hidden_size = hidden_size

        self.fc = torch.nn.Linear(hidden_size, regression_size, bias=False)

    def forward(self, cell_states, target_traces):
        # target_hidden_trace: (B, pedestrian_num, hidden_size)

        regression_list = []
        for i in range(self.pedestrian_num):
            input_attn_hidden_trace = cell_states[:, i]
            target_delta_trace = self.fc(input_attn_hidden_trace)

            regression_list.append(target_delta_trace)
        regression_traces = torch.stack(regression_list, 1)
        regression_traces = regression_traces + target_traces

        # regression_traces: (B, pedestrian_num, regression_size)
        return regression_traces


def initial_fc_weights(modules):
    for m in modules:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)


if __name__ == '__main__':
    torch.manual_seed(0)
    model = EncoderWithLSTM(20, 2, 128)
    model.eval()
    input = torch.rand(32, 20, 2)
    hidden = torch.rand(20, 2, 2, 32, 128)
    output = model(input, hidden)
    print(output[0].size())
