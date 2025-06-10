import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

from DataPrep.AuctionDataset import *

from tqdm import tqdm

def biography_embedding(artists_details_df, device):
    print("Computing representations for artist biography:")
    artists_details_dataset = ArtistDataset(artists_details_df[artists_details_df['artist_biography']!='N/A'])
    artists_details_loader = DataLoader(artists_details_dataset, batch_size=64, shuffle=False)

    biography_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    biography_encoder = BertModel.from_pretrained('bert-base-cased').to(device).eval()

    biography_encoded = {}

    with torch.no_grad():
        loop = tqdm(artists_details_loader, desc="Epoch 1", leave=True)
        for artist_ids, artist_biography in loop:
            inputs = biography_tokenizer(artist_biography, padding=True, truncation=True, return_tensors="pt").to(device)
            x_biography_encoded = biography_encoder(**inputs)
            x_biography_encoded = x_biography_encoded.pooler_output

            biography_encoded.update({artist_id: embedding for artist_id, embedding in zip(artist_ids, x_biography_encoded)})

        input_na = biography_tokenizer(['N/A', 'N/A'], padding=True, truncation=True, return_tensors="pt").to(device)
        na_encoded = biography_encoder(**input_na)
        na_encoded = na_encoded.pooler_output

        biography_encoded.update({'N/A': na_encoded[0]})

    torch.save(biography_encoded, 'Datasets/artist_biography_bert_embedding.pt')

    print(f"Completed | No. of mapping saved: {len(biography_encoded)} | Total No. of artists: {len(artists_details_df)}")

    return biography_encoded

class NeuralNet_w_emb(nn.Module):
    def __init__(self, labelencoders, num_cols, ffn_dim):
        super().__init__()

        # Embeddings for categorical variables
        # self.CAT_EMB_DIM = 50
        self.cat_embeddings = nn.ModuleList([nn.Embedding(len(labelencoder.classes_), len(labelencoder.classes_)//2) for c, labelencoder in labelencoders.items()])
        
        # Dimensions
        self.input_dim = sum([len(labelencoder.classes_)//2 for c, labelencoder in labelencoders.items()]) + len(num_cols)


        # NN
        self.DROP_OUT_RATIO = 0.2
        self.ffn = nn.Sequential()
        for layer_idx, layer_dim in enumerate(ffn_dim):
            input_dim = ffn_dim[layer_idx-1] if layer_idx > 0 else self.input_dim
            self.ffn.append(nn.Sequential(nn.Linear(input_dim, layer_dim), nn.BatchNorm1d(layer_dim), nn.ReLU(), nn.Dropout(self.DROP_OUT_RATIO)))
        self.ffn.append(nn.Sequential(nn.Linear(ffn_dim[-1], 1)))

    def forward(self, x_cat, x_num, x_biography):
        """
        x_cat: (batch_size, num_categorical_predictors)
        x_num: (batch_size, num_numerical_predictors)
        x_biography: tuple of str, length = batch_size 
        """
        # embed x_cat
        x_cat_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat(x_cat_emb + [x_num], dim=1)

        return self.ffn(x)


class NeuralNet_w_emb_res(nn.Module):
    def __init__(self, labelencoders, num_cols, ffn_dim):
        super().__init__()

        # Embeddings for categorical variables
        # self.CAT_EMB_DIM = 50
        self.cat_embeddings = nn.ModuleList([nn.Embedding(len(labelencoder.classes_), len(labelencoder.classes_)//2) for c, labelencoder in labelencoders.items()])
        
        # Dimensions
        self.input_dim = sum([len(labelencoder.classes_)//2 for c, labelencoder in labelencoders.items()]) + len(num_cols)

        # NN
        self.DROP_OUT_RATIO = 0.2
        self.ffn = nn.Sequential()
        for layer_idx, layer_dim in enumerate(ffn_dim):
            input_dim = ffn_dim[layer_idx-1] if layer_idx > 0 else self.input_dim
            self.ffn.append(nn.Sequential(nn.Linear(input_dim, layer_dim), nn.BatchNorm1d(layer_dim), nn.ReLU(), nn.Dropout(self.DROP_OUT_RATIO)))
        self.ffn.append(nn.Sequential(nn.Linear(ffn_dim[-1], 1)))

    def forward(self, x_cat, x_num, x_biography=None, biography_encoded=None):
        """
        x_cat: (batch_size, num_categorical_predictors)
        x_num: (batch_size, num_numerical_predictors)
        x_biography: tuple of str, length = batch_size 
        """
        # embed x_cat
        x_cat_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat(x_cat_emb + [x_num], dim=1)

        return self.ffn(x)+x_num[:,-1].unsqueeze(1)
    

class NeuralNet_w_emb_bert_res(nn.Module):
    def __init__(self, labelencoders, num_cols, ffn_dim):
        super().__init__()

        # Embeddings for categorical variables
        self.cat_embeddings = nn.ModuleList([nn.Embedding(len(labelencoder.classes_), len(labelencoder.classes_)//2) for c, labelencoder in labelencoders.items()])
        
        # CLIP model for encoding biography
        # self.biography_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # self.biography_encoder = BertModel.from_pretrained('bert-base-cased').eval()
        self.biography_encoded_dim = 768

        # embedding for artists without biography
        # self.biography_na_embedding = nn.Embedding(1, self.biography_encoded_dim)

        # Dimensions
        self.input_dim = sum([len(labelencoder.classes_)//2 for c, labelencoder in labelencoders.items()]) + len(num_cols) + self.biography_encoded_dim

        # NN
        self.DROP_OUT_RATIO = 0.2
        self.ffn = nn.Sequential()
        for layer_idx, layer_dim in enumerate(ffn_dim):
            input_dim = ffn_dim[layer_idx-1] if layer_idx > 0 else self.input_dim
            self.ffn.append(nn.Sequential(nn.Linear(input_dim, layer_dim), nn.BatchNorm1d(layer_dim), nn.ReLU(), nn.Dropout(self.DROP_OUT_RATIO)))
        self.ffn.append(nn.Sequential(nn.Linear(ffn_dim[-1], 1)))

    def forward(self, x_cat, x_num, artist_ids, biography_encoded):
        # embed x_cat
        x_cat_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]

        # encode biography
        x_biography = torch.stack([biography_encoded.get(artist_id, biography_encoded['N/A']) for artist_id in artist_ids], dim=0)

        x = torch.cat(x_cat_emb + [x_biography] + [x_num], dim=1)

        return self.ffn(x)+x_num[:,-1].unsqueeze(1)