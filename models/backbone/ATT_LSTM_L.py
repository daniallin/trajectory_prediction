import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderWithLSTM(nn.Module):
    def __init__(self, args):
        super(EncoderWithLSTM, self).__init__()
        # input_size = 2
        self.args = args
        self.pedestrian_num = args.pedestrian_num
        self.input_size = args.input_size
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, dropout=self.dropout)

    def forward(self, input_traces, pre_hiddens):
        """
        :param input_traces: size: (batch, pedestrian_num, 2), where 2 means (x, y)
        :param pre_hiddens: size: [self.pedestrian_num, (2, [self.n_layers, batch_size, self.hidden_size])]
        :return: encoder_traces: (batch, pedestrian_num, hidden_size)
                 next_hiddens, the same size as pre_hiddens
        """
        batch_size = self.args.batch_size
        next_hiddens = []
        encoder_traces = torch.zeros(batch_size, self.pedestrian_num, self.hidden_size)
        for i in range(self.pedestrian_num):
            input_trace = input_traces[:, i, :].unsqueeze(0)
            encoder_trace, next_hidden = self.lstm(input_trace, (pre_hiddens[i][0], pre_hiddens[i][1]))
            encoder_traces[:, i, :] = encoder_trace.squeeze(0)
            next_hiddens.append(next_hidden)

        return encoder_traces, next_hiddens

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True)
        hidden = hidden.cuda() if self.args.use_cuda else hidden
        return [[ hidden for _ in range(2)]
                for _ in range(self.pedestrian_num)]


class Attention(nn.Module):
    def __init__(self, pedestrian_num, hidden_size, method, use_cuda):
        super(Attention, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.hidden_size = hidden_size
        self.method = method
        self.use_cuda = use_cuda
        # Define layers
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attention = nn.ModuleList([
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1)
            ])

    def forward(self, decoder_hiddens, encoder_outputs):
        batch_size = encoder_outputs.size()[0]
        energies = torch.zeros(batch_size, self.pedestrian_num)
        energies = energies.cuda() if self.use_cuda else energies
        for i in range(self.pedestrian_num):
            energies[:, i] = self._score(decoder_hiddens[:, i, :], encoder_outputs[:, i, :])
        return F.softmax(energies, dim=-1)

    def _score(self, hidden, encoder_output):
        """Calculate the relevance of a particular encoder output in respect to the decoder hidden."""

        if self.method == 'dot':
            energy = torch.bmm(hidden.unsqueeze(1), encoder_output.unsqueeze(2)).squeeze(-1)
        elif self.method == 'general':
            energy = self.attention(encoder_output)
            energy = torch.bmm(hidden.unsqueeze(1), energy.unsqueeze(2)).squeeze(-1)
        elif self.method == 'concat':
            energy = self.attention[0](torch.cat((hidden, encoder_output), -1))
            energy = self.attention[1](energy)
            energy = self.attention[2](energy)
        return energy.squeeze(-1)


class DecoderWithAttention(nn.Module):
    def __init__(self, args):
        super(DecoderWithAttention, self).__init__()
        self.args = args
        self.pedestrian_num = args.pedestrian_num
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.att_method = args.att_method
        self.lstm = nn.LSTM(self.hidden_size*2, self.hidden_size, self.n_layers, dropout=self.dropout)
        self.attention = Attention(self.pedestrian_num, self.hidden_size, self.att_method, self.args.use_cuda)
        self.out = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, decoder_inputs, last_context, last_hiddens, encoder_outputs):
        batch_size = decoder_inputs.size()[0]
        next_hiddens = []
        decoder_outputs = torch.zeros(batch_size, self.pedestrian_num, self.hidden_size)
        for i in range(self.pedestrian_num):
            decoder_input = torch.cat((decoder_inputs[:, i, :], last_context[:, i, :]), -1).unsqueeze(0)
            decoder_output, next_hidden = self.lstm(decoder_input, (last_hiddens[i][0], last_hiddens[i][1]))
            decoder_outputs[:, i, :] = decoder_output.squeeze(0)
            next_hiddens.append(next_hidden)

        # Attention
        attention_weights = self.attention(decoder_outputs, encoder_outputs).unsqueeze(-1)  # B*N
        context = attention_weights.expand(encoder_outputs.size()) * encoder_outputs  # B*N*H
        outputs = F.log_softmax(self.out(torch.cat((decoder_outputs, context), 2)), dim=-1)

        return outputs, context, next_hiddens

    def init_input(self, batch_size):
        input = torch.zeros(batch_size, self.pedestrian_num, self.hidden_size, requires_grad=True)
        input = input.cuda() if self.args.use_cuda else input
        return input


class PredictionNet(nn.Module):
    def __init__(self, args):
        super(PredictionNet, self).__init__()
        self.args = args
        self.pedestrian_num = args.pedestrian_num
        self.regression_size = args.target_size  # 2
        self.hidden_size = args.hidden_size

        self.fc = torch.nn.Linear(self.hidden_size, self.regression_size)

    def forward(self, cell_states, target_traces):
        # target_hidden_trace: (B, pedestrian_num, hidden_size)

        regression_list = []
        for i in range(self.pedestrian_num):
            input_attn_hidden_trace = cell_states[:, i]
            target_delta_trace = self.fc(input_attn_hidden_trace)

            regression_list.append(target_delta_trace)
        regression_traces = torch.stack(regression_list, 1)
        regression_traces = regression_traces + target_traces   # B*N*2

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
