# Run once after uv sync, from project root
import urllib.request, pathlib


url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt'
out = pathlib.Path('models/sam2.1_hiera_small.pt')
out.parent.mkdir(exist_ok=True)
print('Downloading SAM 2.1 hiera_small')
urllib.request.urlretrieve(url, out)
print(f'Saved to {out}')
