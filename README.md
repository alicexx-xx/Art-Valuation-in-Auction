# Art-Valuation-in-Auction

Please find a brief summary of the project [here](https://docs.google.com/document/d/1dxsM76TSi0MgAqOlViMw8608Birvqb5hTJ4SHY-BlNM/edit?tab=t.0)

## Web Scraping and Data Processing
The raw data from web scrapping is cleaned using `Data Processing/results_cleaning_concat_v2.py` and output 2 pickle files: `auction_results_cleaned.pickle` and `artists_details.pickle` under folder `Datasets`. We uploaded a sample of the raw data from web scrapping and the two processed data files in this [Google Driver folder](https://drive.google.com/drive/folders/1plR1_Lm5csHVgk4F5LEJEnbZkUP1sua0?usp=drive_link)

The remaining scripts in `Data Processing` should be run in the sequence: `RawData_Processing.ipynb` $\rightarrow$ `DataImputation.ipynb` $\rightarrow$ `DataSplitting.ipynb` to produce the final datasets for modelling.

In `Datasets`, We also included 
* the US CPI data downloaded from https://www.bls.gov/cpi/data.htm in `Datasets`, which is used as a predictor in our modelling
* `Price_Estimates.csv` with the actual price sold and price estimates reported, used for evaluation of manual appraisals in script `Manual Appraisal Evaluation/PriceEstimateRange.ipynb`

## Modelling
* Models without "residual connection" can be found in `Modeling`.
* Models with "residual connection" (i.e. $\hat{Y}=f(\mathbf{X})+X_{AvgHistoricalPrice}$) can be found in `Models_res_connections`
