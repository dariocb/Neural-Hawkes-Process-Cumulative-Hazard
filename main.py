import os
import pathlib
import pickle
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Memory
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from model import NHP

# folder = pathlib.Path(__file__).parent.resolve()

# if os.name == "nt":
#     _location = os.path.join(folder, "cache")
#     if _location.startswith("\\\\"):
#         _location = "\\\\?\\UNC\\" + _location[2:]
#     else:
#         _location = "\\\\?\\" + _location
# else:
#     _location = os.path.join(folder, "cache")

# cache = Memory(location=_location)


class PatientDataset(Dataset):
    def __init__(self, appointments):
        self.patients = appointments.ID.unique()
        self.appointments = appointments
        # self.get_patient_info = cache.cache(self._get_patient_info)
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
            return {"feat": None, "iat": None, "mask": None, "clinic_history": None}

        df["Fecha consulta"] = df["Fecha consulta"].apply(pd.to_datetime)
        iat = (
            df["Fecha consulta"][1:].values - df["Fecha consulta"][:-1].values
        ) / np.timedelta64(1, "D")
        # iat = [elem if elem is not None else torch.tensor([torch.inf]) for elem in iat]
        df = df.set_index(["ID", "Fecha consulta"])

        clinic_history = df[["Antecedentes psiquiátricos", "Antecedentes somáticos"]]
        df = df.drop(["Antecedentes psiquiátricos", "Antecedentes somáticos"], axis=1)

        mask = clinic_history.isna().values

        return {
            "feat": torch.tensor(df.values),
            "iat": torch.log(torch.tensor(iat))
            if iat.size != 0
            else torch.tensor([torch.inf]),
            "mask": torch.tensor(mask),
            "clinic_history": list(map(list, clinic_history.values)),
        }


def custom_collate(batch):
    featU, iatU, maskU, histU, featN, iatN, maskN, histN = zip(*batch)

    def pad_pack(obj, nulls):
        obj = [o if o is not None else nulls for o in obj]
        lengths = [seq.shape[0] for seq in obj]
        obj_padded = pad_sequence(obj, batch_first=True)
        packed_obj = pack_padded_sequence(
            obj_padded.float(),
            [l if l != 0 else 1 for l in lengths],
            batch_first=True,
            enforce_sorted=False,
        )
        return packed_obj

    def pad_2d(batch):
        max_words = 0

        for patient in batch:
            if patient is None:
                patient = []
            for visit in patient:
                visit[0] = np.array([[0]]) if visit[0] is None else visit[0]
                visit[1] = np.array([[0]]) if visit[1] is None else visit[1]
                if isinstance(visit[0], str):
                    visit[0] = visit[1]
                temp = max(visit[0].shape[0], visit[1].shape[0])
                max_words = max(temp, max_words)

        max_t = max(len(seq) if seq is not None else 0 for seq in batch)

        result = np.zeros((len(batch), max_t, 2, max_words, 80))

        for b, patient in enumerate(batch):
            for t in range(max_t):
                for v in range(2):
                    try:
                        seq_len = len(patient[t][v])
                        result[b, t, v, :seq_len, :] = patient[t][v]
                    except (TypeError, IndexError):
                        pass

        return torch.tensor(result, dtype=float).reshape((len(batch), -1, 160))#.to_sparse()

    packed_featU = pad_pack(featU, nulls=torch.empty((0, 35)))
    packed_featN = pad_pack(featN, nulls=torch.empty((0, 35)))
    packed_iatU = pad_pack(iatU, nulls=torch.tensor([torch.inf]))
    packed_iatN = pad_pack(iatN, nulls=torch.tensor([torch.inf]))
    packed_maskU = pad_pack(maskU, nulls=torch.empty((0, 2)))
    packed_maskN = pad_pack(maskN, nulls=torch.empty((0, 2)))
    packed_histU = pad_2d(histU)
    packed_histN = pad_2d(histN)

    return (
        packed_featU,
        packed_iatU,
        packed_maskU,
        packed_histU,
        packed_featN,
        packed_iatN,
        packed_maskN,
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
            data = pickle.load(file)
        print("Done!")
    else:
        data = PatientDataset(merged)
        with open(path, "wb") as file:
            pickle.dump(data, file)

    dataloader = DataLoader(
        data, batch_size=64, collate_fn=custom_collate, shuffle=True, num_workers=20, pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":
    dataloader = get_model_input()

    kwargs = {
        "feat_input_size": 35,
        "feat_hidden_size": 128,
        "feat_n_layers": 2,
        "feat_dropout": 0.2,
        "feat_proj": 12,
        "hist_input_size": 160,
        "hist_hidden_size": 128,
        "hist_n_layers": 2,
        "hist_dropout": 0.2,
        "hist_proj": 12,
        "lr": 0.001,
    }
    model = NHP(**kwargs)

    # Initialize PyTorch Lightning Trainer with early stopping
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=[0],
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True),
            pl.callbacks.EarlyStopping(
                monitor="avg_val_loss",
                patience=3,
                mode="min",
                verbose=True,
                min_delta=0.001,
            ),
        ],
    )

    # Train the model
    trainer.fit(model, dataloader)

    # Plot training loss
    plt.plot(trainer.callback_metrics["avg_train_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.show()
