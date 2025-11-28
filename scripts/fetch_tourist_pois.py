import requests
import time
import pandas as pd
import math
import os

AREAS = {
    "baguio": {"lat_min":16.35, "lat_max":16.45, "lon_min":120.55, "lon_max":120.65},
    "cordillera": {"lat_min":16.0, "lat_max":18.0,   "lon_min":120.0, "lon_max":121.5},
    "luzon": {"lat_min":12.0, "lat_max":19.0,   "lon_min":116.0, "lon_max":122.0},
    "brunei": {"lat_min": 4.0,  "lat_max": 5.0,   "lon_min": 114.0, "lon_max": 115.5},
    "cambodia": {"lat_min": 10.0, "lat_max": 14.7,  "lon_min": 102.3, "lon_max": 107.6},
    "indonesia": {"lat_min": -10.0,"lat_max": 6.0,   "lon_min": 95.0,  "lon_max": 141.0},
    "laos": {"lat_min": 13.5, "lat_max": 22.5,  "lon_min": 100.0, "lon_max": 107.5},
    "malaysia": {"lat_min": 0.85, "lat_max": 7.5,   "lon_min": 99.5,  "lon_max": 119.5},
    "myanmar": {"lat_min": 9.5,  "lat_max": 28.5,  "lon_min": 92.0,  "lon_max": 101.0},
    "philippines": {"lat_min": 4.5,  "lat_max": 20.0,  "lon_min": 116.0,"lon_max": 126.5},
    "singapore": {"lat_min": 1.1,  "lat_max": 1.5,   "lon_min": 103.6,"lon_max": 104.0},
    "thailand": {"lat_min": 5.0,  "lat_max": 20.5,  "lon_min": 97.5, "lon_max": 105.5},
    "vietnam": {"lat_min": 8.0,  "lat_max": 23.5,  "lon_min": 102.0,"lon_max": 110.5}
}

GRID_SIZE = 0.5
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TAGS = [
    ('tourism','attraction'),
    ('tourism','museum'),
    ('tourism','viewpoint'),
    ('tourism','theme_park'),
    ('tourism','zoo'),
    ('tourism','yes'),
    ('historic','monument'),
    ('historic','castle'),
    ('historic','archaeological_site'),
    ('leisure','park'),
    ('leisure','garden'),
    ('leisure','amusement_arcade'),
    ('leisure','common'),
    ('leisure','water_park'),
    ('tourism','gallery'),
    ('leisure','playground')
]

PAUSE_SEC = 1.5
OUT_FOLDER = "data"
os.makedirs(OUT_FOLDER, exist_ok=True)

def build_overpass_query(lat_min, lon_min, lat_max, lon_max, tags):
    tag_queries = []
    for k,v in tags:
        tag_queries.append(f'  node["{k}"="{v}"]({lat_min},{lon_min},{lat_max},{lon_max});')
        tag_queries.append(f'  way["{k}"="{v}"]({lat_min},{lon_min},{lat_max},{lon_max});')
        tag_queries.append(f'  relation["{k}"="{v}"]({lat_min},{lon_min},{lat_max},{lon_max});')
    tag_str = "\n".join(tag_queries)
    query = f"""
    [out:json][timeout:180];
    (
    {tag_str}
    );
    out center;
    """
    return query

def fetch_pois(lat_min, lon_min, lat_max, lon_max, tags):
    query = build_overpass_query(lat_min, lon_min, lat_max, lon_max, tags)
    for attempt in range(3):
        try:
            response = requests.get(OVERPASS_URL, params={'data': query}, timeout=180)
            if response.status_code == 200:
                data = response.json()
                return data.get('elements', [])
            else:
                print(f"Status {response.status_code}, retrying...")
                time.sleep(2 + attempt*2)
        except Exception as e:
            print(f"Attempt {attempt+1} failed:", e)
            time.sleep(2 + attempt*2)
    return []

def process_elements(elements):
    rows = []
    for el in elements:
        tags = el.get('tags', {})
        name = tags.get('name')
        if not name:
            continue
        lat = el.get('lat') or el.get('center', {}).get('lat')
        lon = el.get('lon') or el.get('center', {}).get('lon')
        if lat is None or lon is None:
            continue
        typ = tags.get('tourism') or tags.get('historic') or tags.get('leisure') or "other"
        rows.append({
            'name': name,
            'lat': lat,
            'lon': lon,
            'type': typ,
            'osm_type': el.get('type'),
            'osm_id': el.get('id')
        })
    return rows

def generate_grid(lat_min, lon_min, lat_max, lon_max, grid_size):
    lat_steps = math.ceil((lat_max - lat_min) / grid_size)
    lon_steps = math.ceil((lon_max - lon_min) / grid_size)
    boxes = []
    for i in range(lat_steps):
        for j in range(lon_steps):
            lat0 = lat_min + i * grid_size
            lat1 = min(lat0 + grid_size, lat_max)
            lon0 = lon_min + j * grid_size
            lon1 = min(lon0 + grid_size, lon_max)
            boxes.append((lat0, lon0, lat1, lon1))
    return boxes


# main
if __name__ == "__main__":
    for area_name, bounds in AREAS.items():
        print(f"\n=== Fetching area: {area_name} ===")
        lat_min, lat_max = bounds['lat_min'], bounds['lat_max']
        lon_min, lon_max = bounds['lon_min'], bounds['lon_max']

        if area_name == "baguio_city":
            # Small area: fetch directly
            elements = fetch_pois(lat_min, lon_min, lat_max, lon_max, TAGS)
            print(f"  Raw elements fetched: {len(elements)}")
            rows = process_elements(elements)
        else:
            # Large area: grid-based fetch
            boxes = generate_grid(lat_min, lon_min, lat_max, lon_max, GRID_SIZE)
            all_rows = []
            print(f"  Total grid boxes: {len(boxes)}")
            for idx, (lat0, lon0, lat1, lon1) in enumerate(boxes):
                print(f"    Fetching box {idx+1}/{len(boxes)}: ({lat0},{lon0}) -> ({lat1},{lon1})")
                elements = fetch_pois(lat0, lon0, lat1, lon1, TAGS)
                print(f"      Raw elements fetched: {len(elements)}")
                rows_box = process_elements(elements)
                all_rows.extend(rows_box)
                time.sleep(PAUSE_SEC)
            rows = all_rows

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=['name','lat','lon'])
        out_file = os.path.join(OUT_FOLDER, f"{area_name}_tourist_pois.csv")
        df.to_csv(out_file, index=False)
        print(f"Saved {len(df)} unique POIs to {out_file}")
