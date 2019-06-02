import torch
import torch.nn as nn

from models.backbone.CIDNN import EncoderNetWithLSTM,\
    DecoderNet, RegressionNet, Attention
from models.backbone.SR_LSTM import EncoderNetLSTM, SRMotionGate, PredictionNet
import models.backbone.ATT_LSTM as ATT
import models.backbone.ATT_LSTM_Spatial as ATS


def build_model(args):
    if args.backbone == 'CIDNN':
        return CIDNN_Model(args)
    elif args.backbone == 'SR_LSTM':
        return SRLSTM_Model(args)
    elif args.backbone == 'ATT_LSTM':
        return ATT_LSTM_Model(args)
    elif args.backbone == 'ATT_LSTM_Spatial':
        return ATT_LSTM_Spatial(args)


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
        self.encoder_net = EncoderNetLSTM(args.pedestrian_num, args.input_size, args.hidden_size, args.n_layers)
        self.decoder_net = SRMotionGate(args.pedestrian_num, args.target_size, args.hidden_size)
        self.regression_net = PredictionNet(args.pedestrian_num, args.target_size, args.hidden_size)

    def forward(self, input_traces):
        batch_size = input_traces.size(0)

        target_traces = input_traces[:, :, self.args.input_frame - 1]
        encoder_hidden = self.encoder_net.init_hidden(batch_size)

        # run LSTM in observation frame
        for i in range(self.args.input_frame - 1):
            input_hidden_traces, encoder_hidden, _ = self.encoder_net(input_traces[:, :, i], encoder_hidden)

        prediction_list = []
        for i in range(self.args.target_frame):
            # encode LSTM
            input_hidden_traces, encoder_hidden, input_cell_states = self.encoder_net(target_traces, encoder_hidden)

            refine_cell_states = self.decoder_net(target_traces, input_cell_states, input_hidden_traces)

            # predict next frame traces
            prediction_traces = self.regression_net(refine_cell_states, target_traces)

            # decoder --> location
            target_traces = prediction_traces

            prediction_list.append(prediction_traces)

        regression_traces = torch.stack(prediction_list, 2)

        return regression_traces


class ATT_LSTM_Model(nn.Module):
    def __init__(self, args):
        super(ATT_LSTM_Model, self).__init__()
        self.args = args
        self.encoder_net = ATT.EncoderWithLSTM(args)
        self.decoder_net = ATT.DecoderWithAttention(args)
        self.regression_net = ATT.PredictionNet(args)

    def forward(self, input_traces, teacher_target_traces=None, val_mode=False):
        batch_size = input_traces.size(0)

        target_traces = input_traces[:, :, self.args.input_frame - 1]
        encoder_hiddens = self.encoder_net.init_hidden(batch_size)
        decoder_inputs_init = self.decoder_net.init_decoder_input(batch_size)
        decoder_context = self.decoder_net.init_context(batch_size)

        for i in range(self.args.input_frame - 1):
            _, encoder_hiddens = self.encoder_net(input_traces[:, :, i], encoder_hiddens)

        all_regres_traces = torch.zeros(batch_size, self.args.pedestrian_num, self.args.target_frame, self.args.target_size)
        all_regres_traces = all_regres_traces.cuda() if self.args.use_cuda else all_regres_traces
        decoder_hiddens = encoder_hiddens
        decoder_inputs =target_traces
        for i in range(self.args.target_frame):
            encoder_traces, encoder_hiddens = self.encoder_net(target_traces, encoder_hiddens)
            if not self.args.teacher:
                decoder_outputs, decoder_context, decoder_hiddens = self.decoder_net(
                    decoder_inputs, decoder_context, decoder_hiddens, encoder_traces)
            else:
                decoder_outputs, decoder_context, decoder_hiddens = self.decoder_net(
                    decoder_inputs, decoder_context, decoder_hiddens, encoder_traces)
            regres_traces = self.regression_net(decoder_outputs, target_traces)
            target_traces = regres_traces
            if val_mode or not self.args.teacher:
                decoder_inputs = target_traces
                # decoder_inputs = decoder_inputs_init
            else:
                decoder_inputs = teacher_target_traces[:, :, i]
            all_regres_traces[:, :, i, :] = regres_traces

        return all_regres_traces


class ATT_LSTM_Spatial(nn.Module):
    def __init__(self, args):
        super(ATT_LSTM_Spatial, self).__init__()
        self.args = args
        self.encoder_net = ATS.EncoderWithLSTM(args)
        self.decoder_net = ATS.DecoderWithAttention(args)
        self.attention = ATS.SpatialAffinity(args)
        self.regression_net = ATS.PredictionNet(args)

    def forward(self, input_traces, teacher_target_traces=None, val_mode=False):
        batch_size = input_traces.size(0)

        target_traces = input_traces[:, :, self.args.input_frame - 1]
        encoder_hiddens = self.encoder_net.init_hidden(batch_size)
        decoder_context = self.decoder_net.init_context(batch_size)

        for i in range(self.args.input_frame - 1):
            _, encoder_hiddens = self.encoder_net(input_traces[:, :, i], encoder_hiddens)

        all_regres_traces = torch.zeros(batch_size, self.args.pedestrian_num, self.args.target_frame, self.args.target_size)
        all_regres_traces = all_regres_traces.cuda() if self.args.use_cuda else all_regres_traces
        decoder_hiddens = encoder_hiddens
        decoder_inputs =target_traces
        for i in range(self.args.target_frame):
            encoder_traces, encoder_hiddens = self.encoder_net(target_traces, encoder_hiddens)
            if not self.args.teacher:
                decoder_outputs, decoder_context, decoder_hiddens = self.decoder_net(
                    decoder_inputs, decoder_context, decoder_hiddens, encoder_traces)
            else:
                decoder_outputs, decoder_context, decoder_hiddens = self.decoder_net(
                    decoder_inputs, decoder_context, decoder_hiddens, encoder_traces)
            att_outputs = self.attention(target_traces, decoder_outputs)
            regres_traces = self.regression_net(att_outputs, target_traces)
            target_traces = regres_traces
            if val_mode or not self.args.teacher:
                decoder_inputs = target_traces
                # decoder_inputs = decoder_inputs_init
            else:
                decoder_inputs = teacher_target_traces[:, :, i]
            all_regres_traces[:, :, i, :] = regres_traces

        return all_regres_traces


if __name__ == '__main__':
    from train import create_args
    parser = create_args()
    args = parser.parse_args()

    # model = CIDNN_Model(args)
    # model.eval()
    # input = torch.rand(256, 20, 5, 2)
    # output = model(input)
    # print(output.size())

    # model = SRLSTM_Model(args)
    # model.eval()
    # input = torch.rand(256, 20, 5, 2)
    # output = model(input)
    # print(output.size())

    # model = ATT_LSTM_Model(args)
    # model.eval()
    # input = torch.rand(256, 20, 5, 2)
    # target = torch.rand(256, 20, 5, 2)
    # output = model(input, target)
    # print(output.size())

    model = ATT_LSTM_Spatial(args)
    model.eval()
    input = torch.rand(256, 20, 5, 2)
    target = torch.rand(256, 20, 5, 2)
    output = model(input, target)
    print(output.size())
