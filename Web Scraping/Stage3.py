"""
Stage 3: Auction Entry Parser

This script processes raw HTML `<a>` tags collected in Stage 2 to extract structured
auction result data for each artist. It parses information such as title, medium,
dimensions, auction house, price, estimate, and image URLs.

What it does:
1. **Loads artist auction tag data** from `auction_results_{letter}_fixed.pickle`.
2. For each artist:
   - Parses all stored auction `<a>` tags using BeautifulSoup.
   - Extracts key fields:
     • Title
     • Medium & Dimensions
     • Sale Date, Auction House, Location, Sale Name, Lot Number
     • Price Sold & Estimated
     • Image URL (original + improved quality)
3. **Saves parsed auction results** in `results/auction_results_{letter}.pickle`
   as a list of dictionaries, one per auction entry.

Output format per auction entry:
{
    'Title': "Composition in Blue",
    'Artist ID': "z12",
    'Artist Name': "Jane Doe",
    'Medium': "Oil on canvas",
    'Dimensions': "24 x 36 in",
    'Sale Date': "2020-05-10",
    'Auction House': "Christie’s",
    'Sale Location': "New York",
    'Sale Name': "Contemporary Art Evening Sale",
    'Lot Number': "Lot 18",
    'Price Sold': "$15,000",
    'Price Estimated': "$12,000 – $18,000",
    'Image url ori': "...",
    'Image url better quality': "..."
}

Notes:
- Uses a custom `modify_url()` function to improve image resolution.
- The input `.pickle` file must contain `<a>` tags as strings.
- Designed to be run per letter (e.g., `'z'`), but can be extended easily.

"""
from bs4 import BeautifulSoup
from scraper_utils import read_list_of_dicts_from_appended_pickle, append_list_of_dicts_to_pickle

def parse_auction_entries(all_entries, artist_id, artist_name):
    auction_data = []
    for entry in all_entries:
        soup = BeautifulSoup(str(entry), "html.parser")
        title = soup.select_one('.bxWaGD')
        medium = soup.select('.irDwAE')[0] if len(soup.select('.irDwAE')) > 0 else None
        dimensions = soup.select('.irDwAE')[1] if len(soup.select('.irDwAE')) > 1 else None
        sale_info = soup.select('.irDwAE')[2] if len(soup.select('.irDwAE')) > 2 else None
        sale_date, auction_house = (sale_info.text.split('•') + ['N/A'])[:2] if sale_info else ('N/A', 'N/A')
        sale_name = soup.select('.irDwAE')[6].text.strip() if len(soup.select('.irDwAE')) > 6 else 'N/A'
        lot_number = soup.select('.irDwAE')[7].text.strip() if len(soup.select('.irDwAE')) > 7 else 'N/A'
        sale_location_raw = soup.select('.irDwAE.bbAxnM')[2].text.strip() if len(soup.select('.irDwAE.bbAxnM')) > 2 else 'N/A'
        sale_location = sale_location_raw.split('•')[1].strip() if '•' in sale_location_raw else 'N/A'
        img = entry.find('img')
        image_url = img.get('src') if img else 'N/A'
        modified_url = modify_url(image_url) if img else 'N/A'
        price_sold = soup.select_one('.cMfkJA') or soup.select_one('.lgFNAw')
        price = price_sold.text.strip() if price_sold else 'N/A'
        estimate = soup.select_one('.jEONpp')
        price_est = estimate.text.strip().replace("(est)", "") if estimate else 'N/A'

        auction_data.append({
            'Title': title.text.strip() if title else 'N/A',
            'Artist ID': artist_id,
            'Artist Name': artist_name,
            'Medium': medium.text.strip() if medium else 'N/A',
            'Dimensions': dimensions.text.strip() if dimensions else 'N/A',
            'Sale Date': sale_date.strip(),
            'Auction House': auction_house.strip(),
            'Sale Location': sale_location,
            'Sale Name': sale_name,
            'Lot Number': lot_number,
            'Price Sold': price,
            'Price Estimated': price_est,
            'Image url ori': image_url,
            'Image url better quality': modified_url
        })
    return auction_data


def modify_url(url_ori, height=400, quality=80, resize_to='fit&amp', width=400):
    try:
        config_param = {'height': str(height), 'quality': str(quality), 'resize_to': resize_to, 'width': str(width)}
        http_, url_details = url_ori.split('://')
        url_domain, url_details = url_details.split('?')
        url_details = url_details.split('&')
        for idx, d in enumerate(url_details):
            if d.split('=')[0] == 'src':
                url_details[idx] = d.replace('thumbnail.jpg', 'larger.jpg')
            else:
                url_details[idx] = '{0}={1}'.format(d.split('=')[0], config_param[d.split('=')[0]])
        return f'{http_}://{url_domain}?{"&".join(url_details)}'
    except Exception as e:
        return url_ori


# ------------------------
# Main Stage 3 Parser
# ------------------------

if __name__ == "__main__":
    letter = 'a'

    output_path = f"results/auction_results_{letter}.pickle"
    open(output_path, 'wb').close()

    print(f"\nLoading auction <a> tag data from auction_results_{letter}_fixed.pickle...")
    entries = read_list_of_dicts_from_appended_pickle(f"auction_results_{letter}_fixed.pickle")
    print(f"Loaded {len(entries)} artist entries.\n")

    for i, entry in enumerate(entries):
        artist_id = entry['artist_id']
        artist_name = entry['artist_name']
        auction_tags = entry['auction_tags']

        print(f"Parsing {len(auction_tags)} entries for {artist_name} ({artist_id})...")

        # Deserialize each saved <a> tag into a BeautifulSoup object
        soup_tags = [BeautifulSoup(tag, "html.parser") for tag in auction_tags]
        parsed = parse_auction_entries(soup_tags, artist_id, artist_name)

        print(f"Parsed {len(parsed)} auctions.")

        append_list_of_dicts_to_pickle(parsed, output_path)

    print(f"\nStage 3 complete — parsed data saved to {output_path}")
