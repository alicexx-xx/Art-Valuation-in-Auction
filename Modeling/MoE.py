"""
This script trains a Mixture of Experts neural network to predict log-transformed art auction prices.

Key features:
- Uses categorical embeddings + numerical features
- Implements a Mixture of Experts model with trainable expert networks and a softmax gating network
- Optimizes hyperparameters with Optuna, including:
  - Number of experts
  - Number of hidden layers per expert
  - Hidden layer dimension
  - Batch size
  - Learning rate
  - Number of epochs

Outputs the best validation MAE and corresponding parameters.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import optuna
import matplotlib.pyplot as plt
import shap


# -------------------------
# LOAD DATA
# -------------------------
train_df = pd.read_pickle("Datasets/train_df.pkl")
val_df = pd.read_pickle("Datasets/val_df.pkl")
val_df = val_df[val_df['Log Price'] < 15].copy()
test_df = pd.read_pickle("Datasets/test_df.pkl")


# -------------------------
# VARIABLE SETUP
# -------------------------
target_col = 'Log Price'
numerical_cols = ['Log Area', 'Sale Year', 'Artist Sale Count', 'CPI_US', 'Artist Ordered Median Price']
cat_cols = ['Paint Imputed', 'Material Imputed', 'Auction House', 'Country', 'Birth Period Ordinal', 'Alive Status']

# -------------------------
# ENCODE CATEGORICAL VARIABLES
# -------------------------
cat_vocab_sizes = {}

for col in cat_cols:
    if col == 'Birth Period Ordinal':
        # Already ordinal integers — just make sure no NaNs and consistent indexing
        for df in [train_df, val_df]:
            df[col + '_idx'] = df[col].fillna(-1).astype(int)
        cat_vocab_sizes[col] = max(
            train_df[col + '_idx'].max(),
            val_df[col + '_idx'].max()
        ) + 1
    else:
        # Standard categorical encoding for string features
        full_cat = pd.concat([train_df[col], val_df[col]], axis=0).astype('category')
        categories = full_cat.cat.categories
        for df in [train_df, val_df]:
            df[col] = pd.Categorical(df[col], categories=categories)
            df[col + '_idx'] = df[col].cat.codes.clip(lower=0)
        cat_vocab_sizes[col] = len(categories)

# -------------------------
# DATASET
# -------------------------
class TabularArtPriceDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols, target_col):
        self.cat_data = df[[col + '_idx' for col in cat_cols]].values.astype('int64')
        self.num_data = df[num_cols].values.astype('float32')
        self.targets = df[target_col].values.astype('float32')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'cat': torch.tensor(self.cat_data[idx], dtype=torch.long),
            'num': torch.tensor(self.num_data[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

train_ds = TabularArtPriceDataset(train_df, cat_cols, numerical_cols, target_col)
val_ds = TabularArtPriceDataset(val_df, cat_cols, numerical_cols, target_col)

# -------------------------
# MODEL
# -------------------------
class MixtureOfExperts(nn.Module):
    def __init__(self, cat_vocab_sizes, num_numerical, n_experts=2, hidden_dims=[64], dropout=0.0):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, min(50, (vocab_size + 1) // 2))
            for vocab_size in cat_vocab_sizes.values()
        ])
        self.embedding_dims = [emb.embedding_dim for emb in self.embeddings]
        input_dim = sum(self.embedding_dims) + num_numerical

        self.experts = nn.ModuleList([
            self._build_expert(input_dim, hidden_dims, dropout)
            for _ in range(n_experts)
        ])

        self.gating = nn.Sequential(
            nn.Linear(input_dim, n_experts),
            nn.Softmax(dim=1)
        )

    def _build_expert(self, input_dim, hidden_dims, dropout):
        layers = []
        for i, dim in enumerate(hidden_dims):
            in_dim = input_dim if i == 0 else hidden_dims[i-1]
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        return nn.Sequential(*layers)

    def forward(self, cat_data, num_data):
        embedded = [emb(cat_data[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_out = torch.cat(embedded, dim=1)
        x = torch.cat([cat_out, num_data], dim=1)

        expert_outputs = torch.stack([expert(x).squeeze(1) for expert in self.experts], dim=1)
        gating_weights = self.gating(x)
        return torch.sum(gating_weights * expert_outputs, dim=1)

# -------------------------
# TRAINING FUNCTION
# -------------------------
def train_mixture_of_experts(model, train_loader, val_loader, n_epochs, lr, weight_decay, patience, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



    best_val_mae = float("inf")
    best_model_state = model.state_dict()
    no_improve = 0
    history = {"train_mae": [], "val_mae": []}

    for epoch in range(n_epochs):
        model.train()
        y_train_true, y_train_pred = [], []
        for batch in train_loader:
            cat, num, target = batch['cat'].to(device), batch['num'].to(device), batch['target'].to(device)
            optimizer.zero_grad()
            output = model(cat, num)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            y_train_true.extend(target.cpu().numpy())
            y_train_pred.extend(output.detach().cpu().numpy())
        train_mae = mean_absolute_error(y_train_true, y_train_pred)

        model.eval()
        y_val_true, y_val_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                cat, num, target = batch['cat'].to(device), batch['num'].to(device), batch['target'].to(device)
                output = model(cat, num)
                y_val_true.extend(target.cpu().numpy())
                y_val_pred.extend(output.cpu().numpy())
        val_mae = mean_absolute_error(y_val_true, y_val_pred)

        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        print(f"Epoch {epoch+1:02d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_model_state)
    return model, history

# -------------------------
# OPTUNA HYPERPARAMETER TUNING
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    n_experts = trial.suggest_int("n_experts", 2, 4)
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 32, 256) for i in range(n_layers)]
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_epochs = trial.suggest_int("n_epochs", 30, 100)
    patience = 20

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MixtureOfExperts(
        cat_vocab_sizes, len(numerical_cols),
        n_experts=n_experts, hidden_dims=hidden_dims, dropout=dropout
    )

    model.to(device)

    def train_fn(model, train_loader, val_loader, n_epochs, lr, patience):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_mae = float("inf")
        best_model_state = model.state_dict()
        no_improve = 0
        history = {"train_mae": [], "val_mae": []}

        for epoch in range(n_epochs):
            model.train()
            y_train_true, y_train_pred = [], []
            for batch in train_loader:
                cat, num, target = batch['cat'].to(device), batch['num'].to(device), batch['target'].to(device)
                optimizer.zero_grad()
                output = model(cat, num)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                y_train_true.extend(target.cpu().numpy())
                y_train_pred.extend(output.detach().cpu().numpy())
            train_mae = mean_absolute_error(y_train_true, y_train_pred)

            model.eval()
            y_val_true, y_val_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    cat, num, target = batch['cat'].to(device), batch['num'].to(device), batch['target'].to(device)
                    output = model(cat, num)
                    y_val_true.extend(target.cpu().numpy())
                    y_val_pred.extend(output.cpu().numpy())
            val_mae = mean_absolute_error(y_val_true, y_val_pred)

            history["train_mae"].append(train_mae)
            history["val_mae"].append(val_mae)

            print(f"Epoch {epoch+1:02d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")


            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        model.load_state_dict(best_model_state)
        return model, history

    model, history = train_fn(model, train_loader, val_loader, n_epochs=n_epochs, lr=lr, patience=patience)
    return min(history["val_mae"])



study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best MAE (log):", study.best_value)
print("Best Parameters:", study.best_params)

# -------------------------
# FINAL TRAINING + EVALUATION
# -------------------------
best_params = study.best_params
train_loader = DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=best_params["batch_size"])

hidden_dims = [best_params[f"hidden_dim_{i}"] for i in range(best_params["n_layers"])]

best_model = MixtureOfExperts(
    cat_vocab_sizes,
    len(numerical_cols),
    n_experts=best_params["n_experts"],
    hidden_dims=hidden_dims,
    dropout=best_params["dropout"]
).to(device)

trained_model, history = train_mixture_of_experts(
    best_model, train_loader, val_loader,
    n_epochs=best_params["n_epochs"],
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"],
    patience=20,
    device=device
)

# -------------------------
# SAVE FINAL MODEL
# -------------------------

# Save model weights
torch.save(trained_model.state_dict(), "best_MOE_model.pth")

# Save hyperparameters
import json
with open("best_MOE_hyperparams", "w") as f:
    json.dump(best_params, f)

# Save cat_vocab_sizes
cat_vocab_sizes_serializable = {k: int(v) for k, v in cat_vocab_sizes.items()}
with open("MOE_cat_vocab_sizes.json", "w") as f:
    json.dump(cat_vocab_sizes_serializable, f)

# Save Num and Cat Columns
with open("MOE_feature_info.json", "w") as f:
    json.dump({
        "numerical_cols": numerical_cols,
        "cat_cols": cat_cols,
        "target_col": target_col
    }, f)

# Set model to eval mode
trained_model.eval()

y_val_true_log = []
y_val_pred_log = []

with torch.no_grad():
    for batch in val_loader:
        cat = batch['cat'].to(device)
        num = batch['num'].to(device)
        target = batch['target'].to(device)

        output = trained_model(cat, num)

        y_val_true_log.extend(target.cpu().numpy())
        y_val_pred_log.extend(output.cpu().numpy())

# Convert to numpy arrays
y_val_true_log = np.array(y_val_true_log)
y_val_pred_log = np.array(y_val_pred_log)

# Mask to remove any infinite or large outliers (optional but useful for art price data)
mask = np.isfinite(y_val_true_log) & np.isfinite(y_val_pred_log) & (y_val_true_log < 100) & (y_val_pred_log < 100)

# Convert from log1p to original price
y_val_true_usd = np.exp(y_val_true_log[mask])
y_val_pred_usd = np.exp(y_val_pred_log[mask])

# Compute USD MAE
mae_usd = mean_absolute_error(y_val_true_usd, y_val_pred_usd)
print(f"Final Validation MAE (USD): ${mae_usd:,.2f}")

# -------------------------
# Plot 1: Predicted vs Actual
# -------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_val_true_log, y_val_pred_log, alpha=0.4)
plt.plot([y_val_true_log.min(), y_val_true_log.max()],
         [y_val_true_log.min(), y_val_true_log.max()], 'r--')
plt.xlabel("Price (log)")
plt.ylabel("Price (log)")
plt.title("Predicted vs Actual Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# Plot 2: Residuals
# -------------------------
residuals = y_val_pred_log - y_val_true_log

plt.figure(figsize=(8, 5))
plt.scatter(y_val_true_log, residuals, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Price (log)")
plt.ylabel("Residual (Predicted - Actual)")
plt.title("Residuals vs Actual Price")
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------------
# SHAP VALUES
# -------------------------

background_num = train_df[numerical_cols].values[:100]
background_cat = train_df[[col + '_idx' for col in cat_cols]].values[:100]
background_combined = np.hstack([background_cat, background_num])

val_cat = val_df[[col + '_idx' for col in cat_cols]].values[:200]
val_num = val_df[numerical_cols].values[:200]
val_combined = np.hstack([val_cat, val_num])

def predict_fn(X):
    X = torch.tensor(X, dtype=torch.float32).to(device)
    cat_data = X[:, :len(cat_cols)].long()
    num_data = X[:, len(cat_cols):]
    trained_model.eval()
    with torch.no_grad():
        preds = trained_model(cat_data, num_data).cpu().numpy()
    return preds.flatten()

explainer = shap.KernelExplainer(predict_fn, background_combined)
shap_values = explainer.shap_values(val_combined)

feature_names = [col + "_idx" for col in cat_cols] + numerical_cols
shap.summary_plot(shap_values, val_combined, feature_names=feature_names)
shap.summary_plot(shap_values, val_combined, feature_names=feature_names, plot_type="bar")


