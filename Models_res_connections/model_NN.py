import os
from DataPrep.AuctionDataset import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from Models.NN import *
from Models.Losses import *

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm

import matplotlib.pyplot as plt

def main():
    raw_data_folder = 'Datasets'
    periods_year = [1400, 1600, 1750, 1850, 1950, 1970, 2025]
    birth_periods = [f'before_{y}' for y in periods_year]

    auction_data = AuctionDataset(raw_data_folder, periods_year)
    auction_data.cleaning()
    auction_data.categorical_variables_grouping()

    filter_con = auction_data.auction_results_filtered['Sale Year'].notnull()
    filter_con = filter_con & auction_data.auction_results_filtered['Price Sold USD'].notnull()
    filter_con = filter_con & auction_data.auction_results_filtered['Cumulative Sales'] > 0

    auction_results_modelling = auction_data.auction_results_filtered[filter_con].copy()
    auction_results_modelling.reset_index(drop=True, inplace=True)
    auction_results_modelling['Log Average Historical Price'] = np.log(auction_results_modelling['Average Historical Price'].fillna(1))
    auction_results_modelling['Log Cumulative Sales'] = np.log(auction_results_modelling['Cumulative Sales'])
    auction_results_modelling['Log Price Sold USD'] = np.log(auction_results_modelling['Price Sold USD'])

    cat_cols = ['Paint','Material','Auction House','Sale Location','Country']
    # num_cols = ['Area_log','Cumulative Auction Count','CPI_US','Log Cumulative Sales','Alive_Yes','Alive_No','Prev Bought In','Repeated Sale','Prev Unknown'] + birth_periods + ['Log Average Historical Price']
    num_cols = ['Area_log','Cumulative Auction Count','CPI_US','Log Cumulative Sales','Alive_Yes','Alive_No',] + birth_periods + ['Log Average Historical Price']
    target_col = ['Log Price Sold USD']

    con_train = auction_results_modelling['Sale Year'] <= 2014
    con_val = (auction_results_modelling['Sale Year'] > 2014)&(auction_results_modelling['Sale Year'] <= 2018)
    con_test = (auction_results_modelling['Sale Year'] > 2018)&(auction_results_modelling['Sale Year'] <= 2025)
    # around 74%, 15%, 11%

    # Encode categorical values
    labelencoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(auction_results_modelling.loc[con_train, c])
        labelencoders[c] = le

    # Mean and Std of numerical values (excl. 0/1 indicators)
    standardize_cols = ['Area_log','Cumulative Auction Count','CPI_US','Log Average Historical Price','Log Cumulative Sales']
    standardize_means = auction_results_modelling.loc[con_train, standardize_cols].mean().to_dict()
    standardize_stds = auction_results_modelling.loc[con_train, standardize_cols].std().to_dict()

    # Target Quantiles (train): for weighted MSE
    b_lower = torch.tensor(auction_results_modelling.loc[con_train, target_col].quantile(0.15))
    b_upper = torch.tensor(auction_results_modelling.loc[con_train, target_col].quantile(0.85))

    auction_data_train = AuctionDatasetNN(auction_results_modelling.loc[con_train, cat_cols + ['Artist ID'] + num_cols + target_col], standardize_means, standardize_stds, labelencoders, num_cols)
    auction_data_val = AuctionDatasetNN(auction_results_modelling.loc[con_val, cat_cols + ['Artist ID'] + num_cols + target_col], standardize_means, standardize_stds, labelencoders, num_cols)
    # auction_data_test = AuctionDatasetNN(auction_results_modelling.loc[con_test, cat_cols + ['Artist ID'] + num_cols + target_col], standardize_means, standardize_stds, labelencoders, num_cols)
    auction_data_train_loader = DataLoader(auction_data_train, batch_size=64, shuffle=False)
    auction_data_val_loader = DataLoader(auction_data_val, batch_size=64, shuffle=False)
    # auction_data_test_loader = DataLoader(auction_data_test, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCH = 50
    PATIENCE = 5
    TRAINING_LOSS = 'MSE'
    LAYER_SIZE = [128, 128, 64]
    LR_start = 5e-4

    model = NeuralNet_w_emb_res(labelencoders, num_cols, LAYER_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_start, weight_decay=1e-4)
    schedular = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
    if TRAINING_LOSS == 'WeightedMSE':
        criterion = WeightedMSELoss(b_lower.to(device), b_upper.to(device))
    elif TRAINING_LOSS == 'Quartic':
        criterion = MeanQuarticError(b_lower.to(device), b_upper.to(device))
    else:
        criterion = nn.MSELoss()

    model_name = f'NN_w_emb_res_{TRAINING_LOSS}'

    meta_data = {
        'MODEL':model_name,
        'DATATSET': 'All_dataprep_refined',
        'FFN_LAYER_SIZE': LAYER_SIZE,
        'LR_INITIAL':LR_start,
        'LR_SCHEDULE': 'LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)',
        'LOSS':TRAINING_LOSS
    }
    

    # Rough estimates, but should be very close to tru
    train_mae_log_scale_history = []
    train_mae_ori_scale_history = []
    val_mae_log_scale_history = []
    val_mae_ori_scale_history = []

    epochs_no_improve = 0
    best_val_mae = 9999
    best_model = {}

    try:
        for epoch in range(EPOCH):
            model.train()
            mae_log_scale = []
            mae_ori_scale = []

            loop = tqdm(auction_data_train_loader, desc=f"Epoch [{epoch+1}/{EPOCH}] Train", leave=True)

            for fea_cat, fea_num, artist_biography, log_price in loop:
                fea_cat = fea_cat.to(device)
                fea_num = fea_num.to(device)
                log_price = log_price.to(device)

                optimizer.zero_grad()
                output = model(fea_cat, fea_num, None, None)
                output = output.squeeze()
                loss = criterion(output, log_price)
                loss.backward()
                optimizer.step()

                mae_log_scale.append(mean_absolute_error(log_price.detach().cpu().numpy(), output.detach().cpu().numpy()))
                mae_ori_scale.append(mean_absolute_error(torch.exp(log_price).detach().cpu().numpy(), torch.exp(output).detach().cpu().numpy()))
            
            train_mae_log_scale = np.mean(mae_log_scale)
            # train_mae_ori_scale = np.mean(mae_ori_scale)
            train_mae_log_scale_history.append(train_mae_log_scale)
            # train_mae_ori_scale_history.append(train_mae_ori_scale)

            schedular.step()

            model.eval()
            mae_log_scale_val = []
            mae_ori_scale_val = []

            with torch.no_grad():
                loop_val = tqdm(auction_data_val_loader, desc=f"Epoch [{epoch+1}/{EPOCH}] Test", leave=True)
                for fea_cat, fea_num, artist_biography, log_price in loop_val:
                    fea_cat = fea_cat.to(device)
                    fea_num = fea_num.to(device)
                    log_price = log_price.to(device)

                    output = model(fea_cat, fea_num, None, None)

                    mae_log_scale_val.append(mean_absolute_error(log_price.detach().cpu().numpy(), output.detach().cpu().numpy()))
                    # mae_ori_scale_val.append(mean_absolute_error(torch.exp(log_price).detach().cpu().numpy(), torch.exp(output).detach().cpu().numpy()))
            
            val_mae_log_scale = np.mean(mae_log_scale_val)
            # val_mae_ori_scale = np.mean(mae_ori_scale_val)


            if val_mae_log_scale >= best_val_mae - 0.0005:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
                best_val_mae = val_mae_log_scale
                best_model = {'model_state_dict':model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'metadata':meta_data}

            val_mae_log_scale_history.append(val_mae_log_scale)
            # val_mae_ori_scale_history.append(val_mae_ori_scale)

            # tqdm.write(f"Epoch [{epoch+1}/{EPOCH}] | Train MAE (log scale): {train_mae_log_scale:.5f} | Val MAE (log scale): {val_mae_log_scale:.5f} | Train MAE (ori scale): {train_mae_ori_scale:.5f} | Val MAE (ori scale): {val_mae_ori_scale:.5f}")
            tqdm.write(f"Epoch [{epoch+1}/{EPOCH}] | Train MAE (log scale): {train_mae_log_scale:.5f} | Val MAE (log scale): {val_mae_log_scale:.5f}")

            if epochs_no_improve >= PATIENCE:
                break
    except KeyboardInterrupt:
        # torch.save(best_model, f'Model Checkpoints/all_data_dataprep_refined/{model_name}.pth')
        torch.save(best_model, f'{model_name}.pth')
        plt.plot(val_mae_log_scale_history, label = 'VAL')
        plt.plot(train_mae_log_scale_history, label = 'TRAIN')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('MAE (log scale)')
        # plt.savefig(f'Model Checkpoints/all_data_dataprep_refined/{model_name}.png')
        plt.savefig(f'{model_name}.png')

    # torch.save(best_model, f'Model Checkpoints/all_data_dataprep_refined/{model_name}.pth')
    torch.save(best_model, f'{model_name}.pth')
    plt.plot(val_mae_log_scale_history, label = 'VAL')
    plt.plot(train_mae_log_scale_history, label = 'TRAIN')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MAE (log scale)')
    # plt.savefig(f'Model Checkpoints/all_data_dataprep_refined/{model_name}.png')
    plt.savefig(f'{model_name}.png')

if __name__ == "__main__":
    main()