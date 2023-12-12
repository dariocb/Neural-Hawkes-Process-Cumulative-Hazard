from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision("medium")


class Permute(torch.nn.Module):
    def forward(self, x):
        return x.transpose(-1, -2)


class NHP(pl.LightningModule):
    def __init__(self, **kwargs):
        super(NHP, self).__init__()
        self._permute = Permute()
        self.attn = nn.MultiheadAttention(
            kwargs["attn_dimension"],
            kwargs["attn_heads"],
            dropout=0.2,
            bias=True,
            batch_first=True,
        )

        self.featRNN = nn.LSTM(
            input_size=kwargs["feat_input_size"],
            hidden_size=kwargs["feat_hidden_size"],
            proj_size=kwargs["feat_proj"],
            num_layers=kwargs["feat_n_layers"],
            bias=True,
            batch_first=False,
            dropout=kwargs["feat_dropout"],
            bidirectional=False,
        )

        self.histRNN = nn.LSTM(
            input_size=kwargs["hist_input_size"],
            hidden_size=kwargs["hist_hidden_size"],
            proj_size=kwargs["hist_proj"],
            num_layers=kwargs["hist_n_layers"],
            bias=True,
            batch_first=True,
            dropout=kwargs["hist_dropout"],
            bidirectional=False,
        )
        self.mlp_hazard = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(kwargs["feat_proj"] + kwargs["hist_proj"], 64)),
                    ("permute11", self._permute),
                    ("batch_norm1", nn.BatchNorm1d(64)),
                    ("permute12", self._permute),
                    ("tanh1", nn.Tanh()),
                    ("dropout1", nn.Dropout(kwargs["hazard_dropout"])),
                    ("fc2", nn.Linear(64, 16)),
                    ("permute21", self._permute),
                    ("batch_norm2", nn.BatchNorm1d(16)),
                    ("permute22", self._permute),
                    ("tanh2", nn.Tanh()),
                    ("dropout2", nn.Dropout(kwargs["hazard_dropout"])),
                    ("fc3", nn.Linear(16, 1)),
                    ("softplus", nn.Softplus()),
                ]
            )
        )

        self.out_hazard = nn.Linear(2 * kwargs["attn_dimension"], 1)

        self.iat_rnn = nn.RNN(
            input_size=kwargs["iat_input_size_rnn"],
            hidden_size=kwargs["iat_hidden_size_rnn"],
            num_layers=1,
            nonlinearity="tanh",
            bias=True,
            batch_first=True,
            dropout=0.2,
        )

        self.lr = kwargs["lr"]

        # Custom constraint for positive weights
        self.mlp_hazard.apply(self.init_weights)
        self.out_hazard.apply(self.init_weights)
        self.iat_rnn.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def enforce_positive_weights(self, layer):
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.data = param.data.abs()

    def forward(
        self,
        packed_featU,
        iatU_input,
        packed_histU,
        packed_featN,
        iatN_input,
        packed_histN,
    ):
        featU, _ = pad_packed_sequence(self.featRNN(packed_featU)[0], batch_first=True)
        featN, _ = pad_packed_sequence(self.featRNN(packed_featN)[0], batch_first=True)
        histU, _ = pad_packed_sequence(self.histRNN(packed_histU)[0], batch_first=True)
        histN, _ = pad_packed_sequence(self.histRNN(packed_histN)[0], batch_first=True)

        h_u = torch.cat([featU[:, -1, :], histU[:, -1, :]], dim=1)
        h_n = torch.cat([featN[:, -1, :], histN[:, -1, :]], dim=1)

        iatU, _ = self.iat_rnn(iatU_input)
        iatN, _ = self.iat_rnn(iatN_input)

        cum_hazard_u = self._calculate_cumulative_hazard(
            h_u, given=h_n, iat=iatU, iat_given=iatN
        )
        cum_hazard_n = self._calculate_cumulative_hazard(
            h_n, given=h_u, iat=iatN, iat_given=iatU
        )

        return (
            cum_hazard_u,
            cum_hazard_n,
            (iatU_input, iatN_input),
        )

    def _calculate_cumulative_hazard(self, x, given, iat, iat_given):
        attn_values, _ = self.attn(iat, iat_given, iat_given)

        x = self.mlp_hazard(
            x.unsqueeze(-2).repeat(1, attn_values.shape[-2], 1) + attn_values
        )

        given = self.mlp_hazard(
            given.unsqueeze(-2).repeat(1, attn_values.shape[-2], 1) + attn_values
        )

        return x + given

    def training_step(self, batch, batch_idx):
        (
            packed_featU,
            packed_iatU,
            packed_histU,
            packed_featN,
            packed_iatN,
            packed_histN,
        ) = batch

        cum_hazard_u, cum_hazard_n, grad_terms = self(
            packed_featU,
            packed_iatU,
            packed_histU,
            packed_featN,
            packed_iatN,
            packed_histN,
        )

        loss = self.loss_fn(cum_hazard_u, grad_terms[0]) + self.loss_fn(
            cum_hazard_n, grad_terms[1]
        )
        return loss

    def loss_fn(self, x, grad_term):
        x.retain_grad()
        grad_term.retain_grad()

        # Calculate the gradient of the output w.r.t. input
        x.backward(gradient=torch.ones_like(x, requires_grad=True), retain_graph=True)
        # gradient = torch.autograd.grad(outputs=x, inputs=grad_terms,
        #                             grad_outputs=torch.ones_like(x, requires_grad=True),
        #                             retain_graph=True, create_graph=True)[0]
        # print(grad_term.grad)
        x.grad.zero_()

        B = x.shape[0]
        loss = 0
        for _ in range(B):
            loss -= 1 / B * torch.sum(torch.log(grad_term.grad) - x)
        return loss

    def on_train_batch_end(self, out, batch, batch_idx):
        self.enforce_positive_weights(self.out_hazard)
        self.enforce_positive_weights(self.iat_rnn)
        self.enforce_positive_weights(self.mlp_hazard)

    # def validation_step(self, batch, batch_idx):
    #     # Add validation step logic if needed
    #     pass

    # def on_validation_epoch_end(self, outputs):
    #     avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     self.log("avg_val_loss", avg_val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def early_stopping_criteria(self, min_delta=0.001, patience=3):
        if len(self.validation_losses) >= patience:
            recent_losses = self.validation_losses[-patience:]
            best_loss = min(recent_losses)
            if all(best_loss - loss < min_delta for loss in recent_losses):
                return True
        return False
