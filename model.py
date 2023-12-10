import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class NHP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super(NHP, self).__init__()


        self.lr = lr
        self.loss_fn = nn.PoissonNLLLoss(log_input=False, full=False)

    def forward(self, packed_input):
        packed_output, _ = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output

    def training_step(self, batch, batch_idx):
        packed_featU, packed_iatU, packed_maskU, packed_histU, \
        packed_featN, packed_iatN, packed_maskN, packed_histN = batch

        # Example: pass packed_featU through your model
        output = self(packed_featU)

        # Example: calculate loss
        target = your_target_tensor  # replace with your actual target tensor
        loss = self.loss_fn(output)

        return loss

    def validation_step(self, batch, batch_idx):
        # Add validation step logic if needed
        pass

    def on_validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1),
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

    def early_stopping_criteria(self, min_delta=0.001, patience=3):
        if len(self.validation_losses) >= patience:
            recent_losses = self.validation_losses[-patience:]
            best_loss = min(recent_losses)
            if all(best_loss - loss < min_delta for loss in recent_losses):
                return True
        return False

