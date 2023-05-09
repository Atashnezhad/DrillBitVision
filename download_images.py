#####################################
# Bing
from bing_image_downloader import downloader

downloader.download("pdc bit", limit=200,  output_dir='dataset',
                    adult_filter_off=True, force_replace=False, timeout=120)

# downloader.download("rollercone bit", limit=200, output_dir='dataset',
#                     adult_filter_off=True, force_replace=False, timeout=60)
