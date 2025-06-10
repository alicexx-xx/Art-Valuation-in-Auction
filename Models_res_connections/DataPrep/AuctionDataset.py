from Utils.PickleIO import *
import pandas as pd
import numpy as np
import re
import os

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


def clean_country_year(country_year):
    country, year = '', ''
    yob, yod = np.nan, np.nan

    country_year = re.split(',', country_year.strip(), maxsplit=1)
    if len(country_year) == 2:
        country, year = country_year
    elif len(country_year) == 1:
        if re.search(r'\d{2,}', country_year[0]):
            year = country_year[0]
        else:
            country = country_year[0]
    
    if ('-' in year) or ('–' in year):
        year = re.findall(r'(?:^|\D)(\d+)\s*[–\-]\s*(\d+)(?:$|\D)', year)
        if len(year) == 1:
            year = year[0]
            if len(year) == 2:
                yob, yod = year
                yob = int(yob) if len(yob) == 4 else np.nan
                yod = int(yod) if len(yod) == 4 else np.nan
        
    elif year != '':
        yob = re.findall(r'(?:^|\D)(\d{4})(?:\D|$)',year)
        yob = int(yob[0]) if len(yob) == 1 else np.nan
    
    return pd.Series([country, yob, yod])

class AuctionDataset:
    def __init__(self, data_folder, periods_year):
        auction_result = read_list_of_dicts_from_appended_pickle(f'{data_folder}/auction_results_cleaned.pickle')
        self.auction_results_df = pd.DataFrame(auction_result)
        self.auction_results_df['Sale Date Cleaned'] = pd.to_datetime(self.auction_results_df['Sale Date Cleaned'])
        
        artist_details = read_list_of_dicts_from_appended_pickle(f'{data_folder}/artists_details.pickle')
        self.artists_details_df = pd.DataFrame(artist_details)
        
        self.data_folder = data_folder
        self.periods_year = periods_year

    def artist_birth_country_year(self):

        artists_details_df = self.artists_details_df.copy()

        artists_details_df[['Country','Year of Birth','Year of Death']] = artists_details_df['artist_country_year'].apply(clean_country_year)
        artists_details_df['Country'] = artists_details_df['Country'].replace('','Unknown')

        periods_year = self.periods_year
        birth_periods = [f'before_{y}' for y in periods_year]
        artists_details_df[birth_periods] = 0
        for (i, y) in enumerate(periods_year):
            artists_details_df.loc[(artists_details_df['Year of Birth']<=y), birth_periods[i:]] = 1

        self.artists_details_df = artists_details_df.copy()
        del artists_details_df

        return birth_periods

    def sale_year(self):
        self.auction_results_df['Sale Year'] = self.auction_results_df['Sale Date Cleaned'].dt.year

    def artist_alive_yn(self):
        con1_yes = self.auction_results_df['Sale Year'] < self.auction_results_df['Year of Death']
        con1_no = self.auction_results_df['Sale Year'] > self.auction_results_df['Year of Death']

        con2_yes = self.auction_results_df['Year of Death'].isnull()&(self.auction_results_df['Sale Year'] < self.auction_results_df['Year of Birth'] + 50)
        con2_no = self.auction_results_df['Year of Death'].isnull()&(self.auction_results_df['Sale Year'] > self.auction_results_df['Year of Birth'] + 100)

        self.auction_results_df['Alive_Yes'] = 0
        self.auction_results_df['Alive_No'] = 0

        self.auction_results_df.loc[con1_yes|con2_yes, 'Alive_Yes'] = 1
        self.auction_results_df.loc[con1_no|con2_no, 'Alive_No'] = 1

        self.auction_results_df['Alive_Unknown'] = 1 - self.auction_results_df[['Alive_Yes','Alive_No']].sum(axis=1)

    def painting_area(self):
        self.auction_results_df['Area'] = self.auction_results_df['Area'].round(decimals=0)
        self.auction_results_df['Area'] = self.auction_results_df['Area'].fillna(self.auction_results_df['Area'].median())
        self.auction_results_df['Area_log'] = np.log(self.auction_results_df['Area'])
        self.auction_results_df['Material'] = self.auction_results_df['Material'].str.strip()

    def historical_median(self):
        if os.path.exists('Datasets/hist_median_avail_dict.pickle'):
            hist_median_avail_dict = read_list_of_dicts_from_appended_pickle('Datasets/hist_median_avail_dict.pickle')
            hist_median_avail = pd.DataFrame(hist_median_avail_dict)
        else:
            hist_median = self.auction_results_df[self.auction_results_df['Sale Date Cleaned'].notnull()].sort_values(by=['Artist ID','Sale Date Cleaned'])[['Artist ID','Sale Date Cleaned','Price Sold USD']].copy()
            hist_median.reset_index(drop=True, inplace=True)
            hist_median['Historical Median'] = None

            prev_aid = ''
            prev_date = pd.to_datetime('2030-01-01')
            prev_prices = []

            for i in tqdm(range(len(hist_median))):
                row = hist_median.iloc[i]
                if row['Artist ID'] != prev_aid:
                    prev_prices = []
                else:
                    if row['Sale Date Cleaned'] != prev_date:
                        hist_median.iloc[i, -1] = np.nanmedian(prev_prices)
                
                prev_prices.append(row['Price Sold USD'])
                prev_aid = row['Artist ID']
                prev_date = row['Sale Date Cleaned']

            hist_median_avail = hist_median[hist_median['Historical Median'].notnull()].copy().reset_index(drop=True)
            hist_median_avail['Sale Date Cleaned'] = hist_median_avail['Sale Date Cleaned'].dt.strftime('%Y-%m-%d')
            hist_median_avail_dict = hist_median_avail.to_dict('records')
            append_list_of_dicts_to_pickle(hist_median_avail_dict, 'Datasets/hist_median_avail_dict.pickle')

        return hist_median_avail

    def cumulative_sales(self):
        self.auction_results_df['Work Count'] = self.auction_results_df['Price Sold USD'].notnull().astype(int)
        cum_sales = self.auction_results_df.groupby(['Artist ID','Sale Date Cleaned'], as_index=False)[['Price Sold USD','Work Count']].sum()
        cum_sales = cum_sales.sort_values(by=['Artist ID','Sale Date Cleaned'])
        cum_sales[['Cumulative Sales','Cumulative Auction Count']] = cum_sales.groupby('Artist ID')[['Price Sold USD','Work Count']].cumsum()
        cum_sales['Cumulative Sales'] = cum_sales['Cumulative Sales'] - cum_sales['Price Sold USD']
        cum_sales['Cumulative Auction Count'] = cum_sales['Cumulative Auction Count'] - cum_sales['Work Count']
        cum_sales['Average Historical Price'] = cum_sales['Cumulative Sales']/cum_sales['Cumulative Auction Count']

        self.auction_results_df = self.auction_results_df.drop('Work Count', axis=1)

        return cum_sales

    def filter_auction_results(self):
        con = (self.auction_results_df['Year of Creation'].fillna('9999').astype(int)>=1945)&(self.auction_results_df['Year of Creation'].fillna('9999').astype(int)<=1970)
        
        return self.auction_results_df[con].reset_index(drop=True).copy()

    def prev_bought_in(self):
        con = self.auction_results_df['Title Cleaned'].str.strip().str.lower().isin(['untitled','n/a','sans titre','senza titolo',])==False
        con = con&(self.auction_results_df['Sale Date Cleaned'].notnull())

        self.auction_results_df['Year of Creation 2'] = self.auction_results_df['Year of Creation'].fillna(9999)
        auction_history = self.auction_results_df[con].groupby(['Artist ID','Title Cleaned','Year of Creation 2','Area','Sale Date Cleaned'], as_index=False).agg({'Bought In':'sum', 'Title':'count'}).rename({'Title':'Work Count','Year of Creation 2':'Year of Creation'}, axis=1)
        auction_history = auction_history.sort_values(by=['Artist ID','Title Cleaned','Year of Creation','Area','Sale Date Cleaned'])
        self.auction_results_df.drop('Year of Creation 2', axis=1, inplace=True)

        auction_history['Bought In'] = (auction_history['Bought In']/auction_history['Work Count']).round(decimals=0)
        auction_history[['Prev Artist ID','Prev Title Cleaned','Prev Year of Creation','Prev Area','Prev Sale Date Cleaned','Prev Bought In']] = auction_history[['Artist ID','Title Cleaned','Year of Creation','Area','Sale Date Cleaned','Bought In']].shift(1)

        same_artwork = (auction_history['Prev Artist ID']==auction_history['Artist ID'])
        same_artwork = same_artwork&(auction_history['Prev Title Cleaned']==auction_history['Title Cleaned'])
        same_artwork = same_artwork&(auction_history['Prev Year of Creation']==auction_history['Year of Creation'])
        same_artwork = same_artwork&(auction_history['Prev Area']==auction_history['Area'])
        auction_history['Repeated Sale'] = 0
        auction_history.loc[same_artwork, 'Repeated Sale'] = 1

        return auction_history

    def country_of_birth_grouping(self):
        country_summary = self.auction_results_filtered['Country'].value_counts().reset_index()
        country_summary.columns = ['Country','count']
        country_summary['Rank'] = range(len(country_summary))
        country_summary['Cover'] = country_summary['count'].cumsum()/len(self.auction_results_filtered)

        con = self.auction_results_filtered['Country'].isin(country_summary.head(40)['Country'].tolist()) == False
        self.auction_results_filtered.loc[con, 'Country'] = 'Others'

    def paint_grouping(self):
        self.auction_results_filtered['Paint'] = self.auction_results_filtered['Paint'].fillna('unknown')
        self.auction_results_filtered['Paint'] = self.auction_results_filtered['Paint'].replace('','unknown')
        paint_summary = self.auction_results_filtered['Paint'].value_counts().reset_index()
        paint_summary.columns = ['Paint','count']
        paint_summary['Rank'] = range(len(paint_summary))
        paint_summary['cover'] = paint_summary['count'].cumsum()/paint_summary['count'].sum()

        con = self.auction_results_filtered['Paint'].isin(paint_summary.head(30)['Paint'].tolist())==False
        self.auction_results_filtered.loc[con, 'Paint'] = 'Others'

    def materials_grouping(self):
        self.auction_results_filtered['Material'] = self.auction_results_filtered['Material'].fillna('unknown')
        self.auction_results_filtered['Material'] = self.auction_results_filtered['Material'].replace('','unknown')
        Material_summary = self.auction_results_filtered['Material'].value_counts().reset_index()
        Material_summary.columns = ['Material','count']
        Material_summary['Rank'] = range(len(Material_summary))
        Material_summary['cover'] = Material_summary['count'].cumsum()/Material_summary['count'].sum()

        con = self.auction_results_filtered['Material'].isin(Material_summary.head(25)['Material'].tolist())==False
        self.auction_results_filtered.loc[con, 'Material'] = 'Others'
    
    def sale_location_grouping(self):
        self.auction_results_filtered['Sale Location'] = self.auction_results_filtered['Sale Location'].str.strip().str.lower()
        self.auction_results_filtered['Sale Location'] = self.auction_results_filtered['Sale Location'].replace('','unknown')
        self.auction_results_filtered['Sale Location'] = self.auction_results_filtered['Sale Location'].replace('n/a','unknown')
        sale_location = self.auction_results_filtered['Sale Location'].value_counts().reset_index()
        sale_location.columns = ['Sale Location','count']
        sale_location['Rank'] = range(len(sale_location))
        sale_location['cover'] = sale_location['count'].cumsum()/sale_location['count'].sum()

        con = self.auction_results_filtered['Sale Location'].isin(sale_location.head(35)['Sale Location'].tolist())==False
        self.auction_results_filtered.loc[con, 'Sale Location'] = 'Others'

    def auction_house_grouping(self):
        self.auction_results_filtered['Auction House'] = self.auction_results_filtered['Auction House'].replace('N/A','unknown')
        self.auction_results_filtered['Auction House'] = self.auction_results_filtered['Auction House'].str.strip().str.lower()
        auction_house = self.auction_results_filtered['Auction House'].value_counts().reset_index()
        auction_house.columns = ['Auction House','count']
        auction_house['Rank'] = range(len(auction_house))
        auction_house['cover'] = auction_house['count'].cumsum()/auction_house['count'].sum()

        con = self.auction_results_filtered['Auction House'].isin(auction_house.head(100)['Auction House'].tolist())==False
        self.auction_results_filtered.loc[con, 'Auction House'] = 'Others'

    def us_cpi(self):
        cpi_us = pd.read_csv(f'{self.data_folder}/cpi_us.csv', header=0, index_col=None)
        cpi_us.columns = ['YEAR','CPI']
        cpi_us = cpi_us[cpi_us['YEAR'] >= 1985]
        cpi_us['CPI'] = cpi_us['CPI'].replace('–',np.nan).astype(float)

        cpi_us_1996 = cpi_us.loc[cpi_us['YEAR']==1996, 'CPI'].iloc[0]
        cpi_us['CPI'] = cpi_us['CPI']/cpi_us_1996

        cpi_us_2024 = cpi_us.loc[cpi_us['YEAR']==2024, 'CPI'].iloc[0]

        cpi_us.loc[cpi_us['YEAR']==2025, 'CPI'] = cpi_us_2024 * 1.026
        return cpi_us

    def cleaning(self):
        birth_periods = self.artist_birth_country_year()
        self.auction_results_df = self.auction_results_df.merge(self.artists_details_df[['artist_id','Country','Year of Birth','Year of Death','artist_biography']+birth_periods], how='left', left_on='Artist ID', right_on='artist_id')
        
        self.sale_year()

        self.artist_alive_yn()
        
        self.painting_area()
        
        cum_sales = self.cumulative_sales()
        self.auction_results_df = self.auction_results_df.merge(cum_sales.drop(['Price Sold USD','Work Count'], axis=1), how='left', left_on=['Artist ID','Sale Date Cleaned'], right_on=['Artist ID','Sale Date Cleaned'])

        # hist_median_avail = self.historical_median()
        # hist_median_avail['Sale Date Cleaned'] = pd.to_datetime(hist_median_avail['Sale Date Cleaned'])
        # self.auction_results_df = self.auction_results_df.merge(hist_median_avail.drop('Price Sold USD', axis=1), how='left', left_on=['Artist ID','Sale Date Cleaned'], right_on=['Artist ID','Sale Date Cleaned'])

        self.auction_results_filtered = self.filter_auction_results()
        
        auction_history = self.prev_bought_in()
        self.auction_results_filtered = self.auction_results_filtered.merge(
            auction_history.loc[auction_history['Repeated Sale']==1, ['Artist ID','Title Cleaned','Year of Creation','Area','Sale Date Cleaned','Prev Bought In','Repeated Sale']],
            how='left', left_on=['Artist ID','Title Cleaned','Year of Creation','Area','Sale Date Cleaned'], right_on=['Artist ID','Title Cleaned','Year of Creation','Area','Sale Date Cleaned']
            )
        self.auction_results_filtered['Prev Bought In'] = self.auction_results_filtered['Prev Bought In'].fillna(0)
        self.auction_results_filtered['Repeated Sale'] = self.auction_results_filtered['Repeated Sale'].fillna(0)
        self.auction_results_filtered['Prev Unknown'] = 1 - self.auction_results_filtered['Repeated Sale']

        cpi_us = self.us_cpi()
        self.auction_results_filtered['CPI_US'] = self.auction_results_filtered['Sale Date Cleaned'].dt.year.map(cpi_us.set_index('YEAR')['CPI'].to_dict())


    def categorical_variables_grouping(self):
        self.country_of_birth_grouping()
        self.paint_grouping()
        self.materials_grouping()
        self.sale_location_grouping()
        self.auction_house_grouping()


class ArtistDataset(Dataset):
    def __init__(self, artists_details_df):
        self.df = artists_details_df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.df['artist_id'].iloc[idx], self.df['artist_biography'].iloc[idx]

class AuctionDatasetNN(Dataset):
    def __init__(self, auction_data_modelling, standardize_means, standardize_stds, labelencoders, num_cols):
        """
        Make sure the target is at the last column of df
        """
        auction_data_modelling = auction_data_modelling.reset_index(drop=True)

        # Encode Categorical
        self.labelencoders = labelencoders
        for c, labelencoder in labelencoders.items():
            val_to_idx = {cat: i for i, cat in enumerate(labelencoder.classes_)}
            auction_data_modelling[c] = auction_data_modelling[c].map(val_to_idx)
            if 'Others' in labelencoder.classes_:
                auction_data_modelling[c] = auction_data_modelling[c].fillna(val_to_idx['Others'])
            else:
                auction_data_modelling[c] = auction_data_modelling[c].fillna(val_to_idx['Unknown'])

        # Standardize Numerical
        self.num_cols = num_cols
        self.standardize_means = standardize_means
        self.standardize_stds = standardize_stds
        for c in standardize_means.keys():
            mu = standardize_means[c]
            sigma = standardize_stds[c]
            auction_data_modelling[c] = (auction_data_modelling[c]-mu)/sigma

        self.df = auction_data_modelling.copy()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        fea_cat = torch.tensor(self.df[self.labelencoders.keys()].iloc[idx].values, dtype=torch.long)
        fea_num = torch.tensor(self.df[self.num_cols].iloc[idx].values, dtype=torch.float32)
        artist_id = self.df['Artist ID'].iloc[idx]
        log_price = torch.tensor(self.df['Log Price Sold USD'].iloc[idx], dtype=torch.float32)

        return fea_cat, fea_num, artist_id, log_price
