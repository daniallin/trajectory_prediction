import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from models.CIDNN import EncoderNetWithLSTM,\
    DecoderNet, RegressionNet, Attention
from models.SR_LSTM import EncoderNetLSTM, SRMotionGate, PredictionNet


def build_model(args):
    if args.backbone == 'CIDNN':
        return CIDNN_Model(args)
    elif args.backbone == 'SR_LSTM':
        return SRLSTM_Model(args)


class CIDNN_Model(nn.Module):
    def __init__(self, args):
        super(CIDNN_Model, self).__init__()
        self.args = args

        self.encoder_net = EncoderNetWithLSTM(args.pedestrian_num, args.input_size, args.hidden_size, args.n_layers)
        self.decoder_net = DecoderNet(args.pedestrian_num, args.target_size, args.hidden_size)
        self.regression_net = RegressionNet(args.pedestrian_num, args.target_size, args.hidden_size)
        self.attn = Attention()

    def forward(self, input_traces):
        batch_size = input_traces.size(0)

        target_traces = input_traces[:, :, self.args.input_frame - 1]
        encoder_hidden = self.encoder_net.init_hidden(batch_size)

        # run LSTM in observation frame
        for i in range(self.args.input_frame - 1):
            input_hidden_traces, encoder_hidden = self.encoder_net(input_traces[:, :, i], encoder_hidden)

        regression_list = []

        for i in range(self.args.target_frame):
            # encode LSTM
            input_hidden_traces, encoder_hidden = self.encoder_net(target_traces, encoder_hidden)

            # NN with Attention
            target_hidden_traces = self.decoder_net(target_traces)
            Attn_nn = self.attn(target_hidden_traces, target_hidden_traces)
            c_traces = torch.bmm(Attn_nn, input_hidden_traces)

            # predict next frame traces
            regression_traces = self.regression_net(c_traces, target_hidden_traces, target_traces)

            # decoder --> location
            target_traces = regression_traces

            regression_list.append(regression_traces)

        regression_traces = torch.stack(regression_list, 2)

        return regression_traces


class SRLSTM_Model(nn.Module):
    def __init__(self, args):
        super(SRLSTM_Model, self).__init__()
        self.args = args
        self.encoder = EncoderNetLSTM(args.pedestrian_num, args.input_size, args.hidden_size, args.n_layers)
        self.motion = SRMotionGate(args.pedestrian_num, args.target_size, args.hidden_size)
        self.prediction = PredictionNet(args.pedestrian_num, args.target_size, args.hidden_size)

    def forward(self, input_traces):
        batch_size = input_traces.size(0)

        target_traces = input_traces[:, :, self.args.input_frame - 1]
        encoder_hidden = self.encoder.init_hidden(batch_size)

        # run LSTM in observation frame
        for i in range(self.args.input_frame - 1):
            input_hidden_traces, encoder_hidden, _ = self.encoder(input_traces[:, :, i], encoder_hidden)

        prediction_list = []
        for i in range(self.args.target_frame):
            # encode LSTM
            input_hidden_traces, encoder_hidden, input_cell_states = self.encoder(target_traces, encoder_hidden)

            refine_cell_states = self.motion(target_traces, input_cell_states, input_hidden_traces)

            # predict next frame traces
            prediction_traces = self.prediction(refine_cell_states, target_traces)

            # decoder --> location
            target_traces = prediction_traces

            prediction_list.append(prediction_traces)

        regression_traces = torch.stack(prediction_list, 2)

        return regression_traces


if __name__ == '__main__':
    from train import create_args
    parser = create_args()
    args = parser.parse_args()

    # model = CIDNN_Model(args)
    # model.eval()
    # input = torch.rand(256, 20, 5, 2)
    # output = model(input)
    # print(output.size())

    model = SRLSTM_Model(args)
    model.eval()
    input = torch.rand(256, 20, 5, 2)
    output = model(input)
    print(output.size())
