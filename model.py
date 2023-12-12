import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt
torch.set_float32_matmul_precision('medium')

class NHP(pl.LightningModule):
    def __init__(
        self,
        feat_input_size,
        feat_hidden_size,
        feat_n_layers,
        feat_dropout,
        feat_proj,
        hist_input_size,
        hist_hidden_size,
        hist_n_layers,
        hist_dropout,
        hist_proj,
        lr=0.001,
    ):
        super(NHP, self).__init__()
        self.featRNN = nn.LSTM(
            input_size=feat_input_size,
            hidden_size=feat_hidden_size,
            proj_size=feat_proj,
            num_layers=feat_n_layers,
            bias=True,
            batch_first=False,
            dropout=feat_dropout,
            bidirectional=False,
        )

        self.histRNN = nn.LSTM(
            input_size=hist_input_size,
            hidden_size=hist_hidden_size,
            proj_size=hist_proj,
            num_layers=hist_n_layers,
            bias=True,
            batch_first=True,
            dropout=hist_dropout,
            bidirectional=False,
        )
        self.cumulative_hazard = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(feat_proj + hist_proj, 64)),
            ('batch_norm1', nn.BatchNorm1d(64)),
            ('tanh1', nn.Tanh()),
            ('dropout1', nn.Dropout(dropout_rate)),
            ('fc2', nn.Linear(64, 16)),
            ('batch_norm2', nn.BatchNorm1d(16)),
            ('tanh2', nn.Tanh()),
            ('dropout2', nn.Dropout(dropout_rate)),
            ('fc3', nn.Linear(16, 1)),
            ('softplus', nn.Softplus())
        ]))
        self.lr = lr
        # self.loss_fn = nn.PoissonNLLLoss(log_input=False, full=False)


        # Custom constraint for positive weights
        self.cumulative_hazard.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def enforce_positive_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                param.data = param.data.abs()

    def forward(
        self,
        packed_featU,
        packed_iatU,
        packed_maskU,
        packed_histU,
        packed_featN,
        packed_iatN,
        packed_maskN,
        packed_histN,
    ):
        featU, _ = pad_packed_sequence(
            self.featRNN(packed_featU)[0], batch_first=True
        )
        featN, _ = pad_packed_sequence(
            self.featRNN(packed_featN)[0], batch_first=True
        )
        histU = self.histRNN(packed_histU.float())[0]
        histN = self.histRNN(packed_histN.float())[0]

        h_u = torch.cat([featU[:,-1,:], histU[:,-1,:]], dim=1)
        h_n = torch.cat([featN[:,-1,:], histN[:,-1,:]], dim=1)

        packed_iatU
        packed_iatN


        output = self.cumulative_hazard()


        return output

    def training_step(self, batch, batch_idx):
        (
            packed_featU,
            packed_iatU,
            packed_maskU,
            packed_histU,
            packed_featN,
            packed_iatN,
            packed_maskN,
            packed_histN,
        ) = batch

        # Example: pass packed_featU through your model
        output = self(
            packed_featU,
            packed_iatU,
            packed_maskU,
            packed_histU,
            packed_featN,
            packed_iatN,
            packed_maskN,
            packed_histN,
        )

        # Example: calculate loss
        target = your_target_tensor  # replace with your actual target tensor
        loss = self.loss_fn(output)

        return loss
    def on_train_batch_end(self, out, batch, batch_idx):
        self.enforce_positive_weights()

    def validation_step(self, batch, batch_idx):
        # Add validation step logic if needed
        pass

    def on_validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_val_loss)

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
