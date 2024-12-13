import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw, :-1]  # All columns except the last one
        train_label = input_data[i+tw:i+tw+1, -1]  # The last column
        inout_seq.append((train_seq ,train_label))
    return inout_seq

def train_model(model, train_inout_seq, epochs=150):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

def predict(model, test_inputs, horizon, train_window):
    model.eval()
    for i in range(horizon):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())
    return test_inputs[-horizon:]

def plot_results(actual, predicted):
    plt.figure(figsize=(10,6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.legend()
    plt.show()

# Example usage
def main(df, target_column, horizon, train_window=12):
    df = df.copy()
    df['target'] = df[target_column]
    all_data = df.values.astype(float)
    train_data = all_data[:-horizon]
    test_data = all_data[-horizon:, -1]  # The target variable is the last column

    train_inout_seq = create_inout_sequences(torch.FloatTensor(train_data), train_window)

    model = LSTMModel(input_size=train_data.shape[1] - 1, hidden_layer_size=100, output_size=1)
    train_model(model, train_inout_seq)

    test_inputs = train_data[-train_window:].tolist()
    predictions = predict(model, test_inputs, horizon, train_window)

    r2 = r2_score(test_data, predictions)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))

    print(f'R2: {r2}')
    print(f'RMSE: {rmse}')

    plot_results(test_data, predictions)

# Generate a sample DataFrame
date_rng = pd.date_range(start='1/1/2020', end='1/01/2021', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])
df['var1'] = np.random.randn(len(date_rng))
df['var2'] = np.random.randn(len(date_rng))
df['var3'] = np.random.randn(len(date_rng))
df['target'] = df['var1'] * 0.5 + df['var2'] * 0.3 + df['var3'] * 0.2 + np.random.randn(len(date_rng)) * 0.1
df.set_index('date', inplace=True)

# Assuming df is your DataFrame and 'target' is the column name of the target variable
main(df, 'target', horizon=12, train_window=12)