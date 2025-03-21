import os
from duckduckgo_search import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
from time import sleep

def search_images(keywords, max_images=200): return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')

urls = search_images('bird photos', max_images=1)
urls[0]

# Searching for a bird photo and seeing what kind of result received. Start by getting URLs from a search
urls = search_images('bird photos', max_images=1)
urls[0]

dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)
im = Image.open(dest)
im.to_thumb(256,256)

# Searching for a forest photo and seeing what kind of result received. Start by getting URLs from a search
download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)

# Collect 200 examples of each of "bird" and "forest" photos, and save each group of photos to a different folder
searches = 'forest','bird'
path = Path('bird_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)

# Some photos might not download correctly which could cause model training to fail, removing them
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

# Using fastai DataBlock train and validate model
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=6)

# fastest widely used computer vision model resnet18 to train
learn = vision_learner(dls, resnet18, metrics=error_rate)

# Test model for word
is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")
learn.fine_tune(3)

