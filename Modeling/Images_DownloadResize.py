
import pandas as pd
import numpy as np
import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pandas as pd


# -------------------------
# LOAD DATA
# -------------------------
df_postwar = pd.read_pickle("Datasets/df_postwar.pkl")
df_postwar['Log Price'] = np.log1p(df_postwar['Price Sold USD'])

# -------------------------
# RESIZING IMAGES
# -------------------------

# Setting Up
IMG_SIZE = (224, 224)
df = df_postwar[ # Only keep rows with image URLs
    df_postwar['Image url better quality'].notna() &
    (df_postwar['Image url better quality'].str.lower() != 'n/a')
].copy() 
#df = df.sample(n=100, random_state=42) # Sample 100 rows for testing

output_dir = "Datasets/images_resized"
os.makedirs(output_dir, exist_ok=True)

# Downaloding Images
failed_urls = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    url = row['Image url better quality']
    artwork_id = f"art_{idx:05d}"
    save_path = os.path.join(output_dir, f"{artwork_id}.jpg")

    # Skip if already downloaded
    if os.path.exists(save_path):
        continue

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(save_path, format='JPEG', quality=90)
    except Exception as e:
        failed_urls.append((idx, url, str(e)))

    
# Save failures for review
if failed_urls:
    pd.DataFrame(failed_urls, columns=["Index", "URL", "Error"]).to_csv("failed_image_downloads.csv", index=False)

print(f"Done. Total failures: {len(failed_urls)}.")



