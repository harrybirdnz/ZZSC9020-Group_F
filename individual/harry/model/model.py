#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import os

#global variables
patience = 20
num_epochs = 100
one_hot_features = ['is_summer', 'is_autumn', 'is_winter', 'is_spring',
                    'is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday', 'is_friday', 'is_saturday', 'is_sunday',
                    'is_weekday', 'is_weekend',
                    'is_jan', 'is_feb', 'is_mar', 'is_apr', 'is_may', 'is_jun', 'is_jul', 'is_aug', 'is_sep', 'is_oct', 'is_nov', 'is_dec'
                    ]

#load and prepare data
def prepare_data(params):
    # 1. Load data
    DATA_DIR = '/home/harry/personal/uni/project/data'

    if params["dataset"] == "2016-2019":
        data = pd.read_csv(os.path.join(DATA_DIR, 'processed/processed2.csv'))
        datetimes = pd.to_datetime(data['datetime_au'])
    elif params["dataset"] == "2010-2019":
        data = pd.read_csv(os.path.join(DATA_DIR, 'processed/processed_full.csv'))
        datetimes = pd.to_datetime(data['datetime_au'], dayfirst=True)

    # Select desired features
    data = data[params['features']]

    # Separate one-hot and continuous features
    continuous_features = [f for f in params['features'] if f not in one_hot_features]
    one_hot_feats = [f for f in params['features'] if f in one_hot_features]

    # Scale only continuous features
    scaler = StandardScaler()
    scaled_continuous = scaler.fit_transform(data[continuous_features].values)
    scaled_continuous = torch.FloatTensor(scaled_continuous)

    # Get one-hot features as tensor
    one_hot_tensor = torch.FloatTensor(data[one_hot_feats].values)

    # Concatenate scaled continuous and one-hot features
    scaled_data = torch.cat([scaled_continuous, one_hot_tensor], dim=1)

    # 3. Create sequences and targets
    sequences = []
    targets = []

    for i in range(len(scaled_data) - params["seq_length"]):
        sequences.append(scaled_data[i:i+params["seq_length"]])  # sequence of 7 days
        targets.append(scaled_data[i+params["seq_length"], 0])   # predict next day's demand (column 0)

    sequences = torch.stack(sequences)
    targets = torch.FloatTensor(targets).unsqueeze(1)  # shape: (n, 1)

    return sequences, targets, datetimes, scaler

def create_sequences(data, seq_length):
    """Create sequences of length seq_length from the data"""
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return torch.stack(sequences)

def create_targets(data, seq_length):
    """Create targets for each sequence (the next demand value after the sequence)"""
    targets = []
    for i in range(seq_length, len(data)):
        target = data[i, 0]  # Assuming demand is the first feature
        targets.append(target)
    return torch.stack(targets)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, activation='relu'):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_projected = self.input_projection(src) * math.sqrt(self.d_model)
        src_projected = self.pos_encoder(src_projected)
        encoder_output = self.transformer_encoder(src_projected, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        last_time_step_output = encoder_output[:, -1, :]
        output = self.output_layer(last_time_step_output)
        return output

def train_transformer_model(sequences, targets, input_dim, datetimes, params):
    """Train the Transformer model"""
    datetimes = pd.to_datetime(datetimes)

    if params["train_test_split"] == "80:20":
        X_train, X_val, y_train, y_val = train_test_split(
        sequences, targets, test_size=0.2, shuffle=False
        )
    elif params["train_test_split"] == "prior:2019":
        train_mask = datetimes.dt.year != 2019
        test_mask  = datetimes.dt.year == 2019
        seq_train_mask = train_mask[params['seq_length']:].to_numpy()
        seq_test_mask  = test_mask[params['seq_length']:].to_numpy()
        X_train, y_train = sequences[seq_train_mask], targets[seq_train_mask]
        X_val,   y_val   = sequences[seq_test_mask], targets[seq_test_mask]
    
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor   = torch.FloatTensor(X_val)
    y_train_tensor = torch.FloatTensor(y_train)
    y_val_tensor   = torch.FloatTensor(y_val)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerModel(
        input_dim=input_dim,
        d_model=params["transformer_encoder_layer_params"]['d_model'],
        nhead=params["transformer_encoder_layer_params"]['nhead'],
        num_layers=params["transformer_layer_params"]['num_layers'],
        dim_feedforward=params["transformer_encoder_layer_params"]['dim_feedforward'],
        dropout=params["transformer_encoder_layer_params"]['dropout'],
        activation=params["transformer_encoder_layer_params"]['activation']
    ).to(device)

    # These are pretty standard choices
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break

    # Return train/test splits for postprocess
    return model, train_losses, val_losses, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor

def evaluate_model(model, X_val, y_val, device):
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        predictions = model(X_val_tensor)
        predictions = predictions.cpu().numpy()
    return predictions


def postprocess(model, X_train, y_train, X_test, y_test, scaler, train_losses, val_losses, params, visualise=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get continuous feature count
    continuous_features = [f for f in range(len(params['features'])) if f not in [params['features'].index(f) for f in one_hot_features if f in params['features']]]
    n_cont = len(continuous_features)

    # Train set
    train_predictions = evaluate_model(model, X_train.numpy(), y_train.numpy(), device)
    dummy_train = np.zeros((len(train_predictions), n_cont))
    dummy_train[:, 0] = train_predictions.flatten()
    train_predictions_original = scaler.inverse_transform(dummy_train)[:, 0]
    dummy_train[:, 0] = y_train.numpy().flatten()
    train_targets_original = scaler.inverse_transform(dummy_train)[:, 0]
    train_mse = np.mean((train_predictions_original - train_targets_original) ** 2)
    train_mae = np.mean(np.abs(train_predictions_original - train_targets_original))
    train_rmse = np.sqrt(train_mse)
    train_mape = np.mean(np.abs((train_predictions_original - train_targets_original) / train_targets_original)) * 100

    # Test set
    test_predictions = evaluate_model(model, X_test.numpy(), y_test.numpy(), device)
    dummy_test = np.zeros((len(test_predictions), n_cont))
    dummy_test[:, 0] = test_predictions.flatten()
    test_predictions_original = scaler.inverse_transform(dummy_test)[:, 0]
    dummy_test[:, 0] = y_test.numpy().flatten()
    test_targets_original = scaler.inverse_transform(dummy_test)[:, 0]
    test_mse = np.mean((test_predictions_original - test_targets_original) ** 2)
    test_mae = np.mean(np.abs(test_predictions_original - test_targets_original))
    test_rmse = np.sqrt(test_mse)
    test_mape = np.mean(np.abs((test_predictions_original - test_targets_original) / test_targets_original)) * 100


    results = {
        'train_mse': train_mse,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_mape': train_mape,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_targets_original': test_targets_original,
        'test_predictions_original': test_predictions_original
    }
    return test_mape, results

    return test_mape

def train_model(params):
    input_dim = len(params['features'])
    sequences, targets, datetimes, scaler_X = prepare_data(params)
    # Get train/test splits from train_transformer_model
    model, train_losses, val_losses, X_train, y_train, X_test, y_test = train_transformer_model(
        sequences, targets, input_dim, datetimes, params
    )
    mape, results = postprocess(model, X_train, y_train, X_test, y_test, scaler_X, train_losses, val_losses, params, params['visualise'])
    return mape, results

def median_mape(params):
    # Calculate median MAPE over 5 runs to reduce variance and impact of outliers - more confident estimate of true performance
    results_list = []
    mapes = []
    # sd of 0.08 on 100 runs, therefore average over 5 runs to reduce variance to ~0.04
    for i in range(5):  # Average over 5 runs to reduce variance
        mape, results = train_model(params)
        mapes.append(mape)
        results_list.append(results)
    median_idx = sorted(range(len(mapes)), key=lambda i: mapes[i])[len(mapes) // 2]
    median_mape_val = mapes[median_idx]
    # Visualise only the best (median) model
    visualise_model_results(results_list[median_idx])
    return median_mape_val

def visualise_model_results(results):
    print("Median Model Results after 5 runs:")
    print(f"\nTrain Set Metrics:")
    print(f"MSE: {results['train_mse']:.4f}")
    print(f"MAE: {results['train_mae']:.4f}")
    print(f"RMSE: {results['train_rmse']:.4f}")
    print(f"MAPE: {results['train_mape']:.4f}")

    print(f"\nTest Set Metrics:")
    print(f"MSE: {results['test_mse']:.4f}")
    print(f"MAE: {results['test_mae']:.4f}")
    print(f"RMSE: {results['test_rmse']:.4f}")
    print(f"MAPE: {results['test_mape']:.4f}")

    # Plot results (test set)
    plot_results(results['train_losses'], results['val_losses'], results['test_targets_original'], results['test_predictions_original'])

def plot_results(train_losses, val_losses, y_val, predictions):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Training and validation loss
    axs[0, 0].plot(train_losses, label='Training Loss')
    axs[0, 0].plot(val_losses, label='Validation Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('MSE Loss')
    axs[0, 0].set_title('Training History')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Predictions vs Actuals (first 100 samples)
    axs[0, 1].plot(y_val[:100], label='Actual', alpha=0.7)
    axs[0, 1].plot(predictions[:100], label='Predicted', alpha=0.7)
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Demand')
    axs[0, 1].set_title('Predictions vs Actual (First 100 samples)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Predictions vs Actuals (entire test set)
    axs[1, 0].plot(y_val, label='Actual', alpha=0.7)
    axs[1, 0].plot(predictions, label='Predicted', alpha=0.7)
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Demand')
    axs[1, 0].set_title('Predictions vs Actual (Full Test Set)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Histogram of prediction errors
    errors = predictions - y_val
    axs[1, 1].hist(errors, bins=30, alpha=0.7, color='orange')
    axs[1, 1].set_xlabel('Prediction Error')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title('Histogram of Prediction Errors')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()