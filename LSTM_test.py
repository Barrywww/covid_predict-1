from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from preprocess import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def generate_data_label():
    owid_data, oxf_data, owid_constant_features, owid_variables = prepare_data()
    countries = generate_country_list(owid_data['iso_code'], oxf_data['iso_code'])

    X_train = pd.concat([pd.read_csv("./country_csv/train/" + country + "_train.csv").fillna(0) for country in countries]).drop(columns=['date'])

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    y_train = X_train[:, 0]
    X_train = X_train[:, 1:]

    X_test = pd.concat([pd.read_csv("./country_csv/test/" + country + "_test.csv").fillna(0) for country in countries]).drop(columns=['date'])
    # y_test = np.array(X_test['new_cases'])
    X_test = scaler.transform(X_test)
    y_test = X_test[:, 0]
    X_test = X_test[:, 1:]
    return X_train, y_train, X_test, y_test, scaler


def make_sequences(x, y, seq_length):
    sequences = []
    labels = []

    for i in range(len(x)-seq_length-1):
        sequences.append(x[i: i + seq_length])
        labels.append([y[i + seq_length]])

    # print('----------------------------------------- sequences -----------------------------------------')
    # print(np.array(sequences).shape, np.array(labels).shape)

    sequences = torch.from_numpy(np.array(sequences)).float()
    labels = torch.from_numpy(np.array(labels)).float()
    return sequences, labels


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers=1, output_size=1):
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


def unscale(X_scaled, target, scaler):

    if type(target) == torch.Tensor:
        y_scaled = torch.clone(target).detach().numpy()
    else:
        y_scaled = target.copy().reshape(-1, 1)

    scaled_data = np.concatenate((y_scaled, X_scaled), axis=1)
    unscaled_data = scaler.inverse_transform(scaled_data)
    return torch.from_numpy(unscaled_data[:, 0])


def train_model(model, train_data, train_labels, test_data, test_labels, scaler, num_epochs=1000, learning_rate=1e-06, print_unscale=False):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    y_train_unscaled = make_sequences(X_train, unscale(X_train, y_train, scaler), 7)[1].view(-1)
    y_test_unscaled = make_sequences(X_test, unscale(X_test, y_test, scaler), 7)[1].view(-1)

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for e in range(num_epochs):
        model.reset_hidden_state()

        y_pred = model(train_data)
        loss = loss_fn(y_pred, train_labels)

        with torch.no_grad():
            y_test_pred = model(test_data)
            test_loss = loss_fn(y_test_pred, test_labels)
            y_test_pred_unscaled = unscale(X_test[:X_test.shape[0] - 8], y_test_pred, scaler)
            test_loss_unscaled = loss_fn(y_test_pred_unscaled, y_test_unscaled)

            y_pred_unscaled = unscale(X_train[:X_train.shape[0] - 8], y_pred, scaler)
            unscaled_loss = loss_fn(y_pred_unscaled, y_train_unscaled)
        if print_unscale:
            print(f'epoch: {e:2}; train loss: {unscaled_loss.item():10.8f}; test loss: {test_loss_unscaled.item():10.8f};')
            train_hist[e] = unscaled_loss.item()
            test_hist[e] = test_loss_unscaled.item()
        else:
            print(f'epoch: {e:2}; train loss: {loss.item():10.8f}; test loss: {test_loss.item():10.8f};')
            train_hist[e] = loss.item()
            test_hist[e] = test_loss.item()

        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        y_test_pred = model(test_data)
        y_test_pred_unscaled = unscale(X_test[:X_test.shape[0] - 8], y_test_pred, scaler)
    if print_unscale:
        return model.eval(), train_hist, test_hist, y_test_pred_unscaled
    else:
        return model.eval(), train_hist, test_hist, y_test_pred


X_train, y_train, X_test, y_test, scaler = generate_data_label()

sequences, labels = make_sequences(X_train, y_train, 7)
test_sequences, test_labels = make_sequences(X_test, y_test, 7)

model = LSTM(sequences.shape[2], 200, 7)
print('------------------------------------------- model -------------------------------------------')
print(model)

print('------------------------------------------- train -------------------------------------------')
trained_model, train_losses, test_losses, test_values = train_model(model, sequences, labels, test_sequences, test_labels, scaler, num_epochs=500)


# plt.plot(range(num_epochs), train_losses, label='train loss')
# plt.plot(range(num_epochs), test_losses, label='test loss')
# plt.ylabel('Loss')
# plt.xlabel('epoch')
# plt.show()
y_test = list(y_test)
test_values = list(test_values)
for i in range(0, len(y_test), 62):
    temp = plt.subplot(10, 20, i // 62 + 1)
    temp.plot(y_test[i: i + 61], label='actual')
    temp.plot(test_values[i: i + 61], label='predicted')
    temp.set_yticklabels([])
    temp.set_xticklabels([])
a0 = plt.subplot(11, 1, 10)
a0.plot(train_losses, label='train loss')
a0.plot(test_losses, label='test loss')
plt.show()
