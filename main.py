import os
import pickle
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from model import NHP

torch.backends.cudnn.enabled = False

INF = -1

BATCH_SIZE = 32


class PatientDataset(Dataset):
    def __init__(self, appointments):
        self.patients = appointments.ID.unique()
        self.appointments = appointments

        self.urg, self.no_urg = self.get_patient_info()

    def get_patient_info(self):
        all_urg = []
        all_no_urg = []
        print("Getting patient information from appointment history...")
        for patient in tqdm(self.patients):
            patient_df = self.appointments[self.appointments.ID == patient]
            urg = self.process_one_patient_appointments(
                patient_df[patient_df["urgencias"] == 1].sort_values("Fecha consulta")
            )
            no_urg = self.process_one_patient_appointments(
                patient_df[patient_df["urgencias"] == -1].sort_values("Fecha consulta")
            )
            all_urg.append(urg)
            all_no_urg.append(no_urg)

        assert len(self.patients) == len(all_no_urg)
        assert len(self.patients) == len(all_urg)
        return all_urg, all_no_urg

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        if isinstance(index, int):
            return list(self.urg[index].values()) + list(self.no_urg[index].values())
        return [
            list(itemU.values()) + list(itemN.values())
            for itemU, itemN in zip(self.urg[index], self.no_urg[index])
        ]

    @staticmethod
    def process_one_patient_appointments(df):
        if len(df) == 0:
            return {"feat": None, "iat": None, "clinic_history": None}

        df["Fecha consulta"] = df["Fecha consulta"].apply(pd.to_datetime)
        iat = (
            df["Fecha consulta"][1:].values - df["Fecha consulta"][:-1].values
        ) / np.timedelta64(1, "D")
        df = df.set_index(["ID", "Fecha consulta"])

        for arr in list(
            df[["Antecedentes psiquiátricos", "Antecedentes somáticos"]].values
        ):
            if isinstance(arr[0], str):
                arr[0] = None
            if isinstance(arr[1], str):
                arr[1] = None

            clinic_history = None
            if (
                arr[0] is not None
                and arr[1] is not None
                and len(arr[0]) > 0
                and len(arr[1]) > 0
            ):
                clinic_history = np.concatenate([arr[0], arr[1]], axis=0).tolist()
            elif arr[0] is None and arr[1] is not None and len(arr[1]) > 0:
                clinic_history = arr[1].tolist()
            elif arr[1] is None and arr[0] is not None and len(arr[0]) > 0:
                clinic_history = arr[0].tolist()

        df = df.drop(["Antecedentes psiquiátricos", "Antecedentes somáticos"], axis=1)

        return {
            "feat": df.values.tolist(),
            "iat": np.log(1 + iat).tolist() if iat.size != 0 else [INF],
            "clinic_history": clinic_history,
        }


def pad_pack(obj, nulls, pad=0, marker=False, pack=True):
    lengths = [len(seq) if seq is not None else 0 for seq in obj]

    obj = [
        torch.tensor(o, dtype=torch.float)
        if o is not None
        else torch.tensor(nulls, dtype=torch.float)
        for o in obj
    ]
    obj_padded = pad_sequence(obj, batch_first=True, padding_value=pad).tolist()

    obj_padded = torch.tensor(obj_padded, dtype=torch.float)
    # print(f'Leaf: {obj_padded.is_leaf}')

    if marker:
        obj_padded = torch.tensor(obj_padded.unsqueeze(-1).tolist(), dtype=torch.float)
    if not pack:
        obj_padded = obj_padded.requires_grad_()

    if pack:
        obj_packed = pack_padded_sequence(
            obj_padded,
            [l if l != 0 else 1 for l in lengths],
            batch_first=True,
            enforce_sorted=False,
        )
        return obj_packed
    else:
        return obj_padded


def custom_collate(batch):
    featU, iatU, histU, featN, iatN, histN = zip(*batch)

    packed_featU = pad_pack(featU, nulls=np.zeros((1, 35)).tolist())
    packed_featN = pad_pack(featN, nulls=np.zeros((1, 35)).tolist())

    packed_iatU = pad_pack(iatU, nulls=[-INF], pad=-INF, marker=True, pack=False)
    packed_iatN = pad_pack(iatN, nulls=[-INF], pad=-INF, marker=True, pack=False)

    packed_histU = pad_pack(histU, nulls=np.zeros((1, 80)).tolist())
    packed_histN = pad_pack(histN, nulls=np.zeros((1, 80)).tolist())

    return (
        packed_featU,
        packed_iatU,
        packed_histU,
        packed_featN,
        packed_iatN,
        packed_histN,
    )


def get_model_input():
    features = pd.read_csv("data/prepared_datasets/df_metadata.csv").set_index(
        "Numero episodio"
    )
    eje1 = pd.read_csv("data/prepared_datasets/df_ejeI.csv").set_index(
        "Numero episodio"
    )
    eje2 = pd.read_csv("data/prepared_datasets/df_ejeII.csv").set_index(
        "Numero episodio"
    )
    eje2.columns = ["F2"]
    clinic_history = pickle.load(open("data/prepared_datasets/antecedentes.pkl", "rb"))

    merged = pd.concat([features, eje1, eje2, clinic_history], axis=1)
    merged = merged[
        ~merged.loc[
            :,
            (merged.columns != "Antecedentes psiquiátricos")
            & (merged.columns != "Antecedentes somáticos"),
        ].duplicated()
    ]
    merged = merged.loc[merged["ID"].dropna().index]
    merged = merged.loc[merged["Fecha consulta"].dropna().index]

    cols = ["ejeI", "codigo_motivo_consulta"]
    # Initialize the OneHotEncoder with sparse output
    encoder = OneHotEncoder(sparse_output=False, drop="first")

    # Fit and transform the selected columns
    encoded_data = encoder.fit_transform(merged[cols])

    # Create a DataFrame with the encoded data
    encoded_df = pd.DataFrame(
        encoded_data, index=merged.index, columns=encoder.get_feature_names_out(cols)
    )

    # Concatenate the original DataFrame and the encoded DataFrame
    merged = pd.concat([merged, encoded_df], axis=1)
    merged.drop(columns=cols, inplace=True)

    for c in merged.columns:
        if c != "Antecedentes psiquiátricos" and c != "Antecedentes somáticos":
            values = merged[c].unique()
            if len(values) == 2 and 0 in values and 1 in values:
                merged[c] = merged[c] * 2 - 1
                print(f"Column {c}: {values} -> {values*2-1}")
            elif c != "ID" and c != "Fecha consulta":
                merged[c] = merged[c] / (2 * np.std(merged[c]))
                print(f"Column {c}: {values} -> {merged[c].unique()}")

    path = "./data/prepared_datasets/dataset.pkl"
    if os.path.exists(path):
        print("Loading dataset... ", end="")
        with open(path, "rb") as file:
            train_dataset, val_dataset = pickle.load(file)
        print("Done!")
    else:
        data = PatientDataset(merged)

        # Split the dataset into training and validation sets
        train_size = int(0.8 * len(data))
        val_size = len(data) - train_size
        train_dataset, val_dataset = random_split(data, [train_size, val_size])

        with open(path, "wb") as file:
            pickle.dump((train_dataset, val_dataset), file)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=custom_collate,
        shuffle=True,
        num_workers=20,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=custom_collate,
        shuffle=False,
        num_workers=20,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    train_dataloader, val_dataloader = get_model_input()

    kwargs = {
        "batch_size": BATCH_SIZE,
        "feat_input_size": 35,
        "feat_hidden_size": 128,
        "feat_n_layers": 2,
        "feat_dropout": 0.2,
        "feat_proj": 12,
        "hist_input_size": 80,  # fixed
        "hist_hidden_size": 128,
        "hist_n_layers": 2,
        "hist_dropout": 0.2,
        "hist_proj": 12,
        "lr": 0.001,
        "iat_input_size_rnn": 1,
        "iat_hidden_size_rnn": (12 + 12),  # feat_proj + hist_proj
        "hazard_dropout": 0.2,
        "attn_dimension": (12 + 12),  # feat_proj + hist_proj
        "attn_heads": 2,
    }
    model = NHP(**kwargs)
    wandb_logger = WandbLogger(
        project="NeuralHawkesProcess", name="Exp1", log_model="all"
    )

    # Initialize PyTorch Lightning Trainer with early stopping
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=[1],
        gradient_clip_val=1.,
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True),
            pl.callbacks.EarlyStopping(
                monitor="avg_val_loss",
                patience=5,
                mode="min",
                verbose=True,
                min_delta=0.001,
            ),
            pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
        ],
        logger=wandb_logger,
        inference_mode=False,  # always compute gradients
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
                # ckpt_path = "NeuralHawkesProcess/lftx3nw6/checkpoints/last.ckpt")

    # Plot training loss
    plt.plot(trainer.callback_metrics["avg_train_loss"])
    plt.plot(trainer.callback_metrics["avg_val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.show()
