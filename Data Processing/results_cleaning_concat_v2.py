import pandas as pd
import numpy as np
import pickle
import re
import dateutil.parser

raw_data_path = "Datasets/Raw"


def read_list_of_dicts_from_appended_pickle(filename):
    """Reads lists of dictionaries from a pickle file that was appended to."""
    results = []
    try:
        with open(filename, "rb") as f:
            while True:
                results.extend(pickle.load(f))  # Extend to add all dictionaries
    except EOFError:
        pass
    return results

def append_list_of_dicts_to_pickle(data, filename):
    """Appends a list of dictionaries to a pickle file."""
    try:
        with open(filename, "ab") as f:
            pickle.dump(data, f)
    except FileNotFoundError:
        with open(filename, "wb") as f:
            pickle.dump(data, f)

def remove_duplicates(letter = 'b'):
    print("---------")
    print(f'Letter: {letter}')

    auction_result = read_list_of_dicts_from_appended_pickle(f'{raw_data_path}/auction_results_{letter}.pickle')
    auction_result_df = pd.DataFrame(auction_result)
    artists = read_list_of_dicts_from_appended_pickle(f'{raw_data_path}/artists_details_{letter}.pickle')
    artists_df = pd.DataFrame(artists)
    artists_df.reset_index(drop=True, inplace=True)
    print("no. records before removing duplicated artists: ",len(auction_result_df))
    
    no_id = artists_df.groupby('url', as_index=False)['artist_id'].nunique()
    no_id = no_id[no_id['artist_id']>1]
    dup_id = artists_df[artists_df['url'].isin(no_id['url'].tolist())].copy()
    dup_id = dup_id.drop_duplicates(subset='url', keep='first')
    auction_result_df = auction_result_df[auction_result_df['Artist ID'].isin(dup_id['artist_id'].tolist())==False]
    artists_df = artists_df[artists_df['artist_id'].isin(dup_id['artist_id'].tolist())==False]
    print("no. records after removing duplicated artists: ",len(auction_result_df))

    # check if image URL is available
    auction_result_df['image_avail'] = (auction_result_df['Image url ori']!='N/A').astype(int)
    auction_result_df = auction_result_df.sort_values(by=['Artist ID','Title','Medium','Dimensions','Sale Date','Lot Number','Price Sold','image_avail'], 
                                                      ascending=[True, True, True, True, True, True, True, False])
    auction_result_df = auction_result_df.drop_duplicates(subset=['Artist ID','Title','Medium','Dimensions','Sale Date','Lot Number','Price Sold'], keep='first')
    print("no. records after removing duplicated auctions: ",len(auction_result_df))

    auction_result_df = auction_result_df.drop('image_avail', axis=1)

    return auction_result_df.reset_index(drop=True), artists_df.reset_index(drop=True)


def clean_title(titles):
    titles = titles.replace('','N/A')
    title_year = titles.str.extractall(r'^((?:.|\n)+?)(?:,\s*(\d+))*$').reset_index(drop=False)
    if title_year['match'].max()>0:
        print("ERROR: failed to extract title and year of creation")
        return None
    else:
        return title_year.drop('match', axis=1).set_index('level_0')
    
def clean_medium(medium):
    medium = medium.str.strip().str.lower()
    medium = medium.replace('','N/A')
    medium = medium.str.replace('watercolour','watercolor')
    paint_material = medium.str.extractall(r'(.+?)(?:(?:\s+on\b(.+)$)|$)').reset_index(drop=False)
    if paint_material['match'].max()>0:
        print("ERROR: failed to extract paint and material")
        return None
    else:
        paint_material = paint_material.drop('match', axis=1).set_index('level_0')
        paint_material.columns = ['Paint','Material']
        paint_material['Paint'] = paint_material['Paint'].str.replace('indian ink','india ink')
        paint_material['Paint'] = paint_material['Paint'].apply(lambda x: ", ".join(sorted([m.strip() for m in re.split(',|and', x) if m.strip() !=''])))
        paint_material.loc[paint_material['Paint'].isin(['works','work','N/A']), 'Paint'] = None
        return paint_material


def str_to_float(num):
    result = np.nan
    if ('/' in num) or ('⁄' in num):
        num = num.split(' ')
        if len(num) == 1:
            num_dec = re.split(r'\/|\⁄', num[0].strip())
            if len(num_dec) == 2:
                result = float(num_dec[0].strip())/float(num_dec[1].strip())
        elif len(num) == 2:
            num_int = float(num[0].strip())
            num_dec = num[1].strip()
            num_dec = re.split(r'\/|\⁄', num_dec)
            if len(num_dec) == 2:
                result = num_int + float(num_dec[0].strip())/float(num_dec[1].strip())
    elif num != '':
        result = float(num)
    
    return result

def clean_dim(dimension):
    dim_1, dim_2, dim_3, unit = np.nan, np.nan, np.nan, None
    # first try to extract dimensions in decimal points
    dim = re.findall(r'(?:^|[^0-9\.\/\⁄])(\d+(?:\.\d+)?)\s*(?:x|X|by|×)\s*(\d+(?:\.\d+)?)\s*(?:(?:x|X|by|×)\s*(\d+(?:\.\d+)?)\s*)?([a-z]+)', dimension.strip())
    no_element = [sum([d!='' for d in grp]) for grp in dim]
    if not dim:
        dim = re.findall(r'(?:^|\D\s|[\(\)])((?:\d+\s)?\d+[\/\⁄]\d+)\s*(?:x|X|by|×)\s*((?:\d+\s)?\d+[\/\⁄]\d+)\s*(?:(?:x|X|by|×)\s*((?:\d+\s)?\d+[\/\⁄]\d+)\s*)?([a-z]+)', dimension.strip())
        no_element = [sum([d!='' for d in grp]) for grp in dim]

    if dim and (max(no_element)>0):
        # return dim[np.argmax(no_element)]
        if len(dim[np.argmax(no_element)])==4:
            dim_1, dim_2, dim_3, unit = dim[np.argmax(no_element)]
            # clean the numbers
            dim_1, dim_2, dim_3 = str_to_float(dim_1), str_to_float(dim_2), str_to_float(dim_3)
    
    return pd.Series((dim_1, dim_2, dim_3, unit))

def clean_dim_old(dim):
    dim = dim.str.strip().str.extractall(r'^(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*([a-z]+)').reset_index(drop=False)
    if dim['match'].max()>0:
        print("ERROR: failed to extract dimensions")
        return None
    else:
        dim.columns = ['level_0','match','Dim_0','Dim_1','Dim_unit']
        dim['Dim_0'] = dim['Dim_0'].astype(errors='ignore')
        dim['Dim_1'] = dim['Dim_1'].astype(errors='ignore')
        return dim.drop('match', axis=1).set_index('level_0')

def parse_date(x):
    try:
        x_cleaned = dateutil.parser.parse(x)
        return x_cleaned
    except:
        return pd.NaT
    
def clean_price(prices):
    sold_price = prices.str.extractall(r'.*US\$((?:\d+,)*\d+)$').reset_index(drop=False)
    sold_price.columns = ['level_0','match','price_usd']
    sold_price['price_usd'] = sold_price['price_usd'].str.replace(',','').astype(float)
    if sold_price['match'].max()>0:
        print("ERROR: failed to clean sold price")
        return None
    else:
        return sold_price.set_index('level_0')['price_usd']
    
def auction_house_aliases(auction_house):
    auction_house_start = auction_house.split(' ')[0]
    if auction_house_start == 'Bonhams':
        return 'Bonhams'
    elif auction_house_start == 'Phillips':
        return 'Phillips'
    elif re.match(r'^Millon \& Associes.*', auction_house):
        return 'Millon & Associes'
    elif re.match(r'^Millon \& Robert.*', auction_house):
        return 'Millon & Associes'
    else:
        return auction_house

if __name__ == "__main__":
    letters = ['u','v','w','x','y','z']

    for l in letters:
        auction_result_df, artists_df = remove_duplicates(l)

        # Cleaning Title
        title_year = clean_title(auction_result_df['Title'])
        if title_year is not None:
            auction_result_df[['Title Cleaned','Year of Creation']] = title_year
            auction_result_df['Year of Creation'] = auction_result_df['Year of Creation'].astype(int, errors='ignore')
        else:
            print(f"Error: letter {l} \n")
            continue

        # Clean Medium
        paint_material = clean_medium(auction_result_df['Medium'])
        if paint_material is not None:
            auction_result_df[['Paint','Material']] = paint_material
        else:
            print(f"Error: letter {l} \n")
            continue

        # Clean Dimensions
        auction_result_df[['Dim_1','Dim_2','Dim_3','Dim_unit']] = auction_result_df['Dimensions'].apply(clean_dim)
        unit_conv = {'cm':1**2, 'in':2.54**2, 'mm':0.1**2}
        auction_result_df['Area'] = (auction_result_df['Dim_1']*auction_result_df['Dim_2']*(auction_result_df['Dim_unit'].map(unit_conv))).round(decimals=2)

        # Sale Date
        auction_result_df['Sale Date Cleaned'] = auction_result_df['Sale Date'].apply(parse_date)

        # Price Sold
        sold_price = clean_price(auction_result_df['Price Sold'])
        if sold_price is not None:
            auction_result_df['Price Sold USD'] = sold_price
        else:
            print(f"Error: letter {l} \n")
            continue

        # Indicator: Bought In
        auction_result_df['Bought In'] = (auction_result_df['Price Sold']=='Bought In').astype(int)

        # Auction House
        auction_result_df['Auction House Cleaned'] = auction_result_df['Auction House'].apply(auction_house_aliases)

        # Change Sales date's format to string
        auction_result_df['Sale Date Cleaned'] = auction_result_df['Sale Date Cleaned'].dt.strftime('%Y-%m-%d')

        auction_result = auction_result_df[['Title','Title Cleaned','Year of Creation','Artist ID','Artist Name','Paint','Material','Dimensions','Area','Sale Date Cleaned','Auction House','Sale Location','Sale Name','Lot Number','Price Sold USD','Bought In','Image url ori','Image url better quality']].to_dict('records')
        artists = artists_df.to_dict('records')

        append_list_of_dicts_to_pickle(auction_result, f'Datasets/auction_results_cleaned.pickle')
        append_list_of_dicts_to_pickle(artists, f'Datasets/artists_details.pickle')