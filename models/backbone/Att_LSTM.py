import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torchtext
import nltk
import time
from datetime import timedelta
import numpy as np
from sklearn import metrics


def save_model(model, model_path):
    """Save model."""
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path, use_cuda=False):
    """Load model."""
    map_location = 'cpu'
    if use_cuda and torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location))
    return model

NEG_INF = -10000
TINY_FLOAT = 1e-6


def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """

    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result


def mask_mean(seq, mask=None):
    """Compute mask average on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_mean : torch.float, size [batch, n_channels]
        Mask mean of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    mask_sum = torch.sum(  # [b,msl,nc]->[b,nc]
        seq * mask.unsqueeze(-1).float(), dim=1)
    seq_len = torch.sum(mask, dim=-1)  # [b]
    mask_mean = mask_sum / (seq_len.unsqueeze(-1).float() + TINY_FLOAT)

    return mask_mean


def mask_max(seq, mask=None):
    """Compute mask max on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_max : torch.float, size [batch, n_channels]
        Mask max of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    torch
    mask_max, _ = torch.max(  # [b,msl,nc]->[b,nc]
        seq + (1 - mask.unsqueeze(-1).float()) * NEG_INF,
        dim=1)

    return mask_max


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    Parameters
    ----------
    seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    max_len: int
        The maximum sequence length in a batch.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)

    return mask


class DynamicLSTM(nn.Module):
    """
    Dynamic LSTM module, which can handle variable length input sequence.

    Parameters
    ----------
    input_size : input size
    hidden_size : hidden size
    num_layers : number of hidden layers. Default: 1
    dropout : dropout rate. Default: 0.5
    bidirectional : If True, becomes a bidirectional RNN. Default: False.

    Inputs
    ------
    input: tensor, shaped [batch, max_step, input_size]
    seq_lens: tensor, shaped [batch], sequence lengths of batch

    Outputs
    -------
    output: tensor, shaped [batch, max_step, num_directions * hidden_size],
         tensor containing the output features (h_t) from the last layer
         of the LSTM, for each t.
    """

    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super(DynamicLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, seq_lens):
        # sort input by descending length
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(x, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)

        # pack input
        x_packed = pack_padded_sequence(
            x_sort, seq_lens_sort, batch_first=True)

        # pass through rnn
        y_packed, _ = self.lstm(x_packed)

        # unpack output
        y_sort, length = pad_packed_sequence(y_packed, batch_first=True)

        # unsort output to original order
        y = torch.index_select(y_sort, dim=0, index=idx_unsort)

        return y


class QuoraModel(nn.Module):
    """Model for quora insincere question classification.
    """

    def __init__(self, args):
        super(QuoraModel, self).__init__()

        vocab_size = args["vocab_size"]
        pretrained_embed = args["pretrained_embed"]
        padding_idx = args["padding_idx"]
        embed_dim = 300
        num_classes = 1
        num_layers = 2
        hidden_dim = 50
        dropout = 0.5

        if pretrained_embed is None:
            self.embed = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embed = nn.Embedding.from_pretrained(
                pretrained_embed, freeze=False)
        self.embed.padding_idx = padding_idx

        self.rnn = DynamicLSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout, bidirectional=True)

        self.fc_att = nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim * 6, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_classes)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, word_seq, seq_len):
        # mask
        max_seq_len = torch.max(seq_len)
        mask = seq_mask(seq_len, max_seq_len)  # [b,msl]

        # embed
        e = self.drop(self.embed(word_seq))  # [b,msl]->[b,msl,e]

        # bi-rnn
        r = self.rnn(e, seq_len)  # [b,msl,e]->[b,msl,h*2]

        # attention
        att = self.fc_att(r).squeeze(-1)  # [b,msl,h*2]->[b,msl]
        att = mask_softmax(att, mask)  # [b,msl]
        r_att = torch.sum(att.unsqueeze(-1) * r, dim=1)  # [b,h*2]

        # pooling
        r_avg = mask_mean(r, mask)  # [b,h*2]
        r_max = mask_max(r, mask)  # [b,h*2]
        r = torch.cat([r_avg, r_max, r_att], dim=-1)  # [b,h*6]

        # feed-forward
        f = self.drop(self.act(self.fc(r)))  # [b,h*6]->[b,h]
        logits = self.out(f).squeeze(-1)  # [b,h]->[b]

        return logits


class Trainer(object):
    """Trainer."""

    def __init__(self, **kwargs):
        self.n_epochs = kwargs["epochs"]
        self.batch_size = kwargs["batch_size"]
        self.validate = kwargs["validate"]
        self.save_best_dev = kwargs["save_best_dev"]
        self.use_cuda = kwargs["use_cuda"]
        self.print_every_step = kwargs["print_every_step"]
        self.optimizer = kwargs["optimizer"]
        self.model_path = kwargs["model_path"]
        self.eval_metrics = kwargs["eval_metrics"]

        self._best_accuracy = 0.0

        self.device = 'cpu'
        if torch.cuda.is_available() and self.use_cuda:
            self.device = 'cuda:0'

    def train(self, network, train_data, dev_data=None):
        # transfer model to gpu if available
        network = network.to(self.device)

        # define batch iterator
        train_iter = torchtext.data.Iterator(
            dataset=train_data, batch_size=self.batch_size,
            train=True, shuffle=True, sort=False,
            device=self.device)

        # define Tester over dev data
        if self.validate:
            default_valid_args = {
                "batch_size": max(8, self.batch_size // 10),
                "use_cuda": self.use_cuda}
            validator = Tester(**default_valid_args)

        start = time.time()
        for epoch in range(1, self.n_epochs + 1):
            # turn on network training mode
            network.train()

            # initialize iterator
            train_iter.init_epoch()

            # one forward and backward pass
            self._train_step(
                train_iter, network, start=start,
                n_print=self.print_every_step, epoch=epoch)

            # validation
            if self.validate:
                if dev_data is None:
                    raise RuntimeError(
                        "self.validate is True in trainer, "
                        "but dev_data is None."
                        " Please provide the validation data.")
                eval_results = validator.test(network, dev_data)

                if self.save_best_dev and self.best_eval_result(eval_results):
                    save_model(network, self.model_path)
                    print("Saved better model selected by validation.")

    def _train_step(self, data_iterator, network, **kwargs):
        """Training process in one epoch.
        """
        step = 0
        for batch in data_iterator:
            (text, text_len), target = batch.text, batch.target

            self.optimizer.zero_grad()
            logits = network(text, text_len)
            loss = network.loss(logits, target.float())
            loss.backward()
            self.optimizer.step()

            if kwargs["n_print"] > 0 and step % kwargs["n_print"] == 0:
                end = time.time()
                diff = timedelta(seconds=round(end - kwargs["start"]))
                print_output = "[epoch: {:>3} step: {:>4}]" \
                    " train loss: {:>4.6} time: {}".format(
                        kwargs["epoch"], step, loss.item(), diff)
                print(print_output)

            step += 1

    def best_eval_result(self, eval_results):
        """Check if the current epoch yields better validation results.

        :param eval_results: dict, format {metrics_name: value}
        :return: bool, True means current results on dev set is the best.
        """
        assert self.eval_metrics in eval_results, \
            "Evaluation doesn't contain metrics '{}'." \
            .format(self.eval_metrics)

        accuracy = eval_results[self.eval_metrics]
        if accuracy > self._best_accuracy:
            self._best_accuracy = accuracy
            return True
        else:
            return False


class Tester(object):
    """Tester."""

    def __init__(self, **kwargs):
        self.batch_size = kwargs["batch_size"]
        self.use_cuda = kwargs["use_cuda"]
        self.device = 'cpu'
        if torch.cuda.is_available() and self.use_cuda:
            self.device = 'cuda:0'

    def test(self, network, dev_data, threshold=0.33):
        # transfer model to gpu if available
        network = network.to(self.device)

        # turn on the testing mode; clean up the history
        network.eval()
        output_list = []
        truth_list = []

        # define batch iterator
        data_iter = torchtext.data.Iterator(
            dataset=dev_data, batch_size=self.batch_size,
            train=False, device=self.device, sort=False)

        # predict
        for batch in data_iter:
            text, target = batch.text, batch.target

            with torch.no_grad():
                prediction = network(*text)

            output_list.append(prediction.detach())
            truth_list.append(target.detach())

        # evaluate
        eval_results = self.evaluate(output_list, truth_list, threshold)
        print("[tester] {}".format(self.print_eval_results(eval_results)))

        return eval_results

    def evaluate(self, predict, truth, threshold=0.33):
        """Compute evaluation metrics.

        :param predict: list of Tensor
        :param truth: list of dict
        :param threshold: threshold of positive probability
        :return eval_results: dict, format {name: metrics}.
        """
        y_trues, y_preds = [], []
        for y_true, logit in zip(truth, predict):
            y_pred = (torch.sigmoid(logit) > threshold).long().cpu().numpy()
            y_true = y_true.cpu().numpy()
            y_trues.append(y_true)
            y_preds.append(y_pred)
        y_true = np.concatenate(y_trues, axis=0)
        y_pred = np.concatenate(y_preds, axis=0)

        precision = metrics.precision_score(y_true, y_pred, pos_label=1)
        recall = metrics.recall_score(y_true, y_pred, pos_label=1)
        f1 = metrics.f1_score(y_true, y_pred, pos_label=1)

        metrics_dict = {"precision": precision, "recall": recall, "f1": f1}

        return metrics_dict

    def print_eval_results(self, results):
        """Override this method to support more print formats.
        :param results: dict, (str: float) is (metrics name: value)
        """
        return ", ".join(
            [str(key) + "=" + "{:.4f}".format(value)
             for key, value in results.items()])


class Predictor(object):
    """An interface for predicting outputs based on trained models.
    """

    def __init__(self, batch_size=8, use_cuda=False):
        self.batch_size = batch_size
        self.use_cuda = use_cuda

        self.device = 'cpu'
        if torch.cuda.is_available() and self.use_cuda:
            self.device = 'cuda:0'

    def predict(self, network, data, threshold=0.33):
        # transfer model to gpu if available
        network = network.to(self.device)

        # turn on the testing mode; clean up the history
        network.eval()
        batch_output = []

        # define batch iterator
        data_iter = torchtext.data.Iterator(
            dataset=data, batch_size=self.batch_size,
            train=False, device=self.device, sort=False)

        for batch in data_iter:
            text = batch.text

            with torch.no_grad():
                prediction = network(*text)

            batch_output.append(prediction.detach())

        return self._post_processor(batch_output, threshold)

    def _post_processor(self, batch_output, threshold=0.33):
        """Convert logit tensor to label."""
        y_preds = []
        for logit in batch_output:
            y_pred = (torch.sigmoid(logit) > threshold).long().cpu().numpy()
            y_preds.append(y_pred)
        y_pred = np.concatenate(y_preds, axis=0)

        return y_pred
