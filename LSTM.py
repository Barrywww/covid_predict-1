from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from preprocess import *
import torch
import torch.nn as nn

owid_data, oxf_data, owid_constant_features, owid_variables = prepare_data()
countries = generate_country_list(owid_data['iso_code'], oxf_data['iso_code'])

X_train = pd.concat([pd.read_csv("./country_csv/train/" + country + "_train.csv").fillna(0) for country in countries]).drop(columns=['date'])

scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
y_train = X_train[:, 0]
X_train = X_train[:, 1:]


def make_sequences(x, y, seq_length):
    sequences = []
    labels = []

    for i in range(len(x)-seq_length-1):
        sequences.append(x[i: i + seq_length])
        labels.append([y[i + seq_length]])

    print('----------------------------------------- sequences -----------------------------------------')
    print(np.array(sequences).shape, np.array(labels).shape)

    return np.array(sequences), np.array(labels)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, output_size=1):
        super().__init__()

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

        self.hidden_cell = (torch.zeros(self.num_layers, self.sequence_length, self.hidden_size), torch.zeros(self.num_layers, self.sequence_length, self.hidden_size))
        self.memory = []

    def reset_hidden_state(self):
        self.hidden_cell = (torch.zeros(self.num_layers, self.sequence_length, self.hidden_size), torch.zeros(self.num_layers, self.sequence_length, self.hidden_size))

    def forward(self, sequence):
        lstm_out, self.hidden_cell = self.lstm(sequence.reshape(len(sequence), self.sequence_length, -1), self.hidden_cell)
        last_time_step = lstm_out.reshape(self.sequence_length, len(sequence), self.hidden_size)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred


def train_model(model, train_data, train_labels):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 60
    for e in range(num_epochs):
        model.reset_hidden_state()
        y_pred = model(train_data)
        loss = loss_fn(y_pred, train_labels)
        loss.backward()
        optimizer.step()
        print(f'epoch: {e:2} loss: {loss.item():10.8f}')


model = LSTM(42, 100, 7, 1)
sequences, labels = make_sequences(X_train, y_train, 7)
sequences = torch.from_numpy(sequences).float()
labels = torch.from_numpy(labels).float()
print('------------------------------------------- model -------------------------------------------')
print(model)

print('------------------------------------------- train -------------------------------------------')
train_model(model, sequences, labels)
