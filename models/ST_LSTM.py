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


class SRMotionGate(nn.Module):
    def __init__(self, pedestrian_num, target_size, hidden_size, L=2):
        super(SRMotionGate, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.L = L

        hidden1_size = 32
        hidden2_size = 64

        self.r_fc1 = torch.nn.Linear(target_size, hidden1_size)
        self.r_fc2 = torch.nn.Linear(hidden1_size, hidden2_size)
        self.r_fc3 = torch.nn.Linear(hidden2_size, hidden_size)

        self.u_fc = torch.nn.Linear(hidden_size*3, 1, bias=False)

        self.g_fc = torch.nn.Linear(hidden_size*3, hidden_size)

        self.c_fc = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        initial_fc_weights(self.modules())

    def forward(self, input_traces, cell_states, hidden_states):
        # input_traces: (batch, pedestrian_num, 2)
        # hidden_states: (batch, pedestrian_num, hidden_size)
        for i in range(self.L):
            motion_cell_states = []
            for p in range(self.pedestrian_num):
                hidden_locations = self.cal_r(input_traces, input_traces[:, p, :])
                encoder_u_list = []
                motion_gate = []
                for l in range(self.pedestrian_num):
                    cat_states = torch.cat((hidden_locations[:, l, :], hidden_states[:, p, :], hidden_states[:, l, :]), 1)
                    encoder_u_p = self.u_fc(cat_states)
                    encoder_u_list.append(encoder_u_p)

                    motion_gate_p = torch.sigmoid(self.g_fc(cat_states))
                    motion_gate.append(motion_gate_p)
                encoder_u_list = torch.stack(encoder_u_list, 2)
                alpha_p = self.cal_alpha(encoder_u_list)
                motion_gate = torch.stack(motion_gate, 1)

                motion_gate_out = torch.mul(motion_gate, hidden_states)

                motion_cell_state = self.c_fc(torch.bmm(alpha_p, motion_gate_out)[:, 0, :])
                motion_cell_state = motion_cell_state + cell_states[:, p, :]
                motion_cell_states.append(motion_cell_state)
            motion_cell_states = torch.stack(motion_cell_states, 1)
            cell_states = motion_cell_states

            # TODO: update hidden states using output gate
        return cell_states

    def cal_r(self, input_traces, target_trace):
        hidden_list = []
        for j in range(self.pedestrian_num):
            relative_location = target_trace - input_traces[:, j, :]
            hidden_location = F.relu(self.r_fc1(relative_location))
            hidden_location = F.relu(self.r_fc2(hidden_location))
            hidden_location = self.r_fc3(hidden_location)

            hidden_list.append(hidden_location)

        # stack all person
        hidden_locations = torch.stack(hidden_list, 1)
        return hidden_locations

    def cal_alpha(self, encoder_u):
        exp_u = torch.exp(encoder_u)
        sum_exp_u = torch.sum(exp_u, 2)
        alpha = []
        for i in range(self.pedestrian_num):
            alpha_j = exp_u[:, :, i] / sum_exp_u
            alpha.append(alpha_j)

        alpha = torch.stack(alpha, 2)
        return alpha


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
