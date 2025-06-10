
# -------------------------
# Neural Network for Tabular Data Only
# -------------------------

"""k
This script trains a neural network model on tabular data to predict the log-transformed price of artworks sold at auction.
It performs the following steps:

1. Loads pre-split training, validation, and test datasets.
2. Encodes categorical variables and scales numerical features.
3. Defines a PyTorch Dataset class and dataloaders for structured data.
4. Builds a neural network with embedding layers for categorical inputs and fully connected layers for prediction.
5. Uses Optuna to optimize hyperparameters including network depth, hidden layer sizes, activation functions,
   dropout rate, learning rate, batch size, and number of epochs.
6. Trains the best model using early stopping and evaluates its performance using MAE in both log scale and USD.
7. Plots predicted vs actual log prices, residuals, and computes SHAP values for feature importance.

This model architecture is designed for structured (non-image) features only.
"""

## Libraries ##
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import copy
import optuna
import copy
import shap


# -------------------------
# LOAD PRE-SPLIT DATASETS
# -------------------------
train_df = pd.read_pickle("Datasets/train_df.pkl")
val_df = pd.read_pickle("Datasets/val_df.pkl")
test_df = pd.read_pickle("Datasets/test_df.pkl")

# -------------------------
# VARIABLE SETUP
# -------------------------
target_col = 'Log Price'
numerical_cols = ['Log Area', 'Artist Sale Count', 'CPI_US', 'Artist Ordered Avg Price', 'Artist Cumulative Price']
cat_cols = ['Paint Imputed', 'Material Imputed', 'Auction House', 'Country', 'Birth Period Ordinal', 'Alive Status']

# Encode Categorical Variables
cat_vocab_sizes = {}
for col in cat_cols:
    if col == 'Birth Period Ordinal':
        for df in [train_df, val_df, test_df]:
            df[col + '_idx'] = df[col].fillna(-1).astype(int)
        cat_vocab_sizes[col] = max(train_df[col + '_idx'].max(), val_df[col + '_idx'].max(), test_df[col + '_idx'].max()) + 1
    else:
        full_cat = pd.concat([train_df[col], val_df[col], test_df[col]], axis=0).astype('category')
        categories = full_cat.cat.categories
        for df in [train_df, val_df, test_df]:
            df[col] = pd.Categorical(df[col], categories=categories)
            df[col + '_idx'] = df[col].cat.codes.clip(lower=0)
        cat_vocab_sizes[col] = len(categories)


# Normalize Numerical Features
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
val_df[numerical_cols] = scaler.transform(val_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])


# -------------------------
# DATASET & DATALOADER
# -------------------------
class ArtPriceTabularDataset(Dataset):
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

train_ds = ArtPriceTabularDataset(train_df, cat_cols, numerical_cols, target_col)
val_ds = ArtPriceTabularDataset(val_df, cat_cols, numerical_cols, target_col)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# -------------------------
# MODEL WITH HYPERPARAMS
# -------------------------
class ArtPriceTabularNN(nn.Module):
    def __init__(self, cat_vocab_sizes, num_numerical, hidden_sizes=[128, 64], dropout_rate=0.3, activation_fn="relu"):
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.embedding_dims = []
        self.activation_fn = activation_fn

        for vocab_size in cat_vocab_sizes.values():
            emb_dim = min(50, (vocab_size + 1) // 2)
            self.embeddings.append(nn.Embedding(vocab_size, emb_dim))
            self.embedding_dims.append(emb_dim)

        self.total_emb_dim = sum(self.embedding_dims)

        layers = []
        in_dim = self.total_emb_dim + num_numerical
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            if activation_fn == "relu":
                layers.append(nn.ReLU())
            elif activation_fn == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation_fn == "elu":
                layers.append(nn.ELU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_size

        layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, cat_data, num_data):
        embedded = [emb(cat_data[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_out = torch.cat(embedded, dim=1)
        x = torch.cat([cat_out, num_data], dim=1)
        return self.fc(x).squeeze()

# -------------------------
# TRAINING FUNCTION
# -------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=50, patience=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    epochs_no_improve = 0
    history = {"train_mae": [], "val_mae": []}

    for epoch in range(n_epochs):
        model.train()
        y_train_true, y_train_pred = [], []

        for batch in train_loader:
            cat = batch['cat'].to(device)
            num = batch['num'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            output = model(cat, num)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            y_train_true.extend(target.detach().cpu().numpy())
            y_train_pred.extend(output.detach().cpu().numpy())

        train_mae = mean_absolute_error(y_train_true, y_train_pred)

        model.eval()
        y_val_true, y_val_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                cat = batch['cat'].to(device)
                num = batch['num'].to(device)
                target = batch['target'].to(device)

                output = model(cat, num)
                y_val_true.extend(target.cpu().numpy())
                y_val_pred.extend(output.cpu().numpy())

        val_mae = mean_absolute_error(y_val_true, y_val_pred)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_model_wts)
    return model, history, best_model_wts

# -------------------------
# OPTUNA OBJECTIVE FUNCTION
# -------------------------
def objective(trial):
    # Tune number of layers and size per layer
    n_layers = trial.suggest_int("n_layers", 3, 8)
    hidden_sizes = [trial.suggest_int(f"hidden_size_{i}", 32, 256) for i in range(n_layers)]

    # Tune activation
    activation_fn = trial.suggest_categorical("activation_fn", ["relu", "leaky_relu", "elu"])

    # Other hyperparameters
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # Tune batch size and epochs
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_epochs = trial.suggest_int("n_epochs", 30, 100)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = ArtPriceTabularNN(
        cat_vocab_sizes=cat_vocab_sizes,
        num_numerical=len(numerical_cols),
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout,
        activation_fn=activation_fn
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model, history, _ = train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=n_epochs)
    return history["val_mae"][-1]


# -------------------------
# RUN OPTUNA
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("Best MAE:", study.best_value)
print("Best Params:", study.best_params)

# -------------------------
# FINAL MODEL TRAINING WITH BEST PARAMETERS
# -------------------------

# Unpack best parameters
best_params = study.best_params

# Rebuild hidden layer sizes
hidden_sizes = [
    best_params[f"hidden_size_{i}"]
    for i in range(best_params["n_layers"])
]

# Rebuild DataLoaders
train_loader = DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=best_params["batch_size"])

# Build model
model = ArtPriceTabularNN(
    cat_vocab_sizes=cat_vocab_sizes,
    num_numerical=len(numerical_cols),
    hidden_sizes=hidden_sizes,
    dropout_rate=best_params["dropout"],
    activation_fn=best_params["activation_fn"]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
criterion = nn.MSELoss()

# Train best model
model, history, best_model_wts = train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=best_params["n_epochs"])
model.load_state_dict(best_model_wts)

# -------------------------
# PREDICT ON VALIDATION SET
# -------------------------

model.eval()
y_val_true = []
y_val_pred = []

with torch.no_grad():
    for batch in val_loader:
        cat = batch["cat"].to(device)
        num = batch["num"].to(device)
        target = batch["target"].to(device)

        output = model(cat, num)
        y_val_true.extend(target.cpu().numpy())
        y_val_pred.extend(output.cpu().numpy())

# Convert to arrays
y_val_true = np.array(y_val_true)
y_val_pred = np.array(y_val_pred)

# Removing largest two predicted values
top2_idx = np.argsort(y_val_pred)[-2:]
mask = np.ones_like(y_val_pred, dtype=bool)
mask[top2_idx] = False
y_val_true_filtered = y_val_true[mask]
y_val_pred_filtered = y_val_pred[mask]


mae_log_filtered = mean_absolute_error(y_val_true_filtered, y_val_pred_filtered)
print(f"Validation MAE (log scale, filtered): {mae_log_filtered:.4f}")

# Convert to USD and compute MAE there
y_val_true_usd = np.expm1(y_val_true_filtered)
y_val_pred_usd = np.expm1(y_val_pred_filtered)
mae_usd_filtered = mean_absolute_error(y_val_true_usd, y_val_pred_usd)
print(f"Validation MAE (USD, filtered): ${mae_usd_filtered:,.2f}")

residuals = y_val_true_filtered - y_val_pred_filtered


# -------------------------
# PREDICT ON TEST SET
# -------------------------

# Create Dataset and DataLoader
test_ds = ArtPriceTabularDataset(test_df, cat_cols, numerical_cols, target_col)
test_loader = DataLoader(test_ds, batch_size=best_params["batch_size"])

# Run Inference
model.eval()
y_test_true = []
y_test_pred = []

with torch.no_grad():
    for batch in test_loader:
        cat = batch["cat"].to(device)
        num = batch["num"].to(device)
        target = batch["target"].to(device)

        output = model(cat, num)
        y_test_true.extend(target.cpu().numpy())
        y_test_pred.extend(output.cpu().numpy())

# Convert to arrays
y_test_true = np.array(y_test_true)
y_test_pred = np.array(y_test_pred)

# Remove top 2 predicted values to avoid outliers
top2_idx = np.argsort(y_test_pred)[-2:]
mask = np.ones_like(y_test_pred, dtype=bool)
mask[top2_idx] = False
y_test_true_filtered = y_test_true[mask]
y_test_pred_filtered = y_test_pred[mask]

# Log MAE
mae_log_test = mean_absolute_error(y_test_true_filtered, y_test_pred_filtered)
print(f"Test MAE (log scale, filtered): {mae_log_test:.4f}")

# Convert to USD and compute MAE
y_test_true_usd = np.expm1(y_test_true_filtered)
y_test_pred_usd = np.expm1(y_test_pred_filtered)
mae_usd_test = mean_absolute_error(y_test_true_usd, y_test_pred_usd)
print(f"Test MAE (USD, filtered): ${mae_usd_test:,.2f}")

# -------------------------
# PLOT: PREDICTED VS ACTUAL
# -------------------------

plt.figure(figsize=(6, 6))
plt.scatter(y_val_true, y_val_pred, alpha=0.5)
plt.plot([y_val_true.min(), y_val_true.max()], [y_val_true.min(), y_val_true.max()], '--', color='gray')
plt.title('Predicted vs Actual Log Prices')
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# PLOT: RESIDUALS
# -------------------------

plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Residuals (Actual - Predicted)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------------
# SHAP VALUES
# ------------------------


background_num = train_df[numerical_cols].values[:100]
background_cat = train_df[[col + '_idx' for col in cat_cols]].values[:100]

background_num_tensor = torch.tensor(background_num, dtype=torch.float32).to(device)
background_cat_tensor = torch.tensor(background_cat, dtype=torch.long).to(device)

def predict_fn(X):
    
    X = torch.tensor(X, dtype=torch.float32).to(device)
    cat_data = X[:, :len(cat_cols)].long()
    num_data = X[:, len(cat_cols):]
    model.eval()
    with torch.no_grad():
        preds = model(cat_data, num_data).cpu().numpy()
    return preds.flatten()


background_combined = np.hstack([background_cat, background_num])


val_cat = val_df[[col + '_idx' for col in cat_cols]].values[:200]
val_num = val_df[numerical_cols].values[:200]
val_combined = np.hstack([val_cat, val_num])


explainer = shap.KernelExplainer(predict_fn, background_combined)

shap_values = explainer.shap_values(val_combined)

# Plot
feature_names = [col + "_idx" for col in cat_cols] + numerical_cols
shap.summary_plot(shap_values, val_combined, feature_names=feature_names)
shap.summary_plot(shap_values, val_combined, feature_names=feature_names, plot_type="bar")



