# pip install earthaccess fsspec s3fs tqdm
#yacinekhaldi
#L%8RPr?KcVUtpVx
import earthaccess as ea
from tqdm import tqdm
import os

ea.login()

# Example: VIIRS/JPSS1 Vegetation Indices 8-day 500m (VJ113A4N)
results = ea.search_data(
    short_name="VJ113A4N",   # VNP18A1/VNP18A2/VJ118A1/VJ118A2/MAIAC, etc.
    temporal=("2015-01-01","2024-12-31"),
    bounding_box=(-9,18,12,38),  # W,S,E,N
)
print("Granules:", len(results))

os.makedirs("viirs_vi", exist_ok=True)
files = ea.download(results, "./datasets/viirs_vi")
print(files)
