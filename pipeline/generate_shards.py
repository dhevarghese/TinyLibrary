import os
import os.path
import glob
import argparse
import tqdm
import webdataset as wds

parser = argparse.ArgumentParser("""Generate sharded dataset from library data.""")
parser.add_argument("--maxcount", type=int, default=50000)
parser.add_argument("--shards", default="./shards", help="directory where shards are written") 
parser.add_argument("--data", default="./data", help="directory containing the data")
args = parser.parse_args()
os.makedirs(args.shards, exist_ok=True)

def count_pages(directory):
    total = 0
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            num_files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            total += num_files
    return total

def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()

def extract_book_and_page(image_path):
    # Get the filename only (without the directory path)
    filename = os.path.basename(image_path)
    # Remove the extension (.jpg)
    filename_without_ext = os.path.splitext(filename)[0]
    # Split the filename into book_name and page_name
    book_name, page_name = filename_without_ext.rsplit('_', 1)
    return book_name, page_name

def make_shard(shard_path, img_paths):
    print("Making shard:", shard_path)
    with wds.TarWriter(shard_path) as sink:
        for page_path in img_paths:
            book_name, page_number = extract_book_and_page(page_path)
            key = f"{book_name}/{page_number}"  
            image = readfile(page_path)
            sample = {"__key__": key, "jpg": image}  # Construct a sample.
            sink.write(sample) 

def write_dataset(data_path, base, nsamples=1000):
    print("Sample count: ", nsamples)
    pages = sorted(glob.glob(data_path + "/*.jpg"))
    
    shard_counter = 0  # Counter for the shard name
    for i in tqdm.tqdm(range(0, len(pages), nsamples), desc="Processing pages"):
        page_chunk = pages[i:i+nsamples]
        shard_name = os.path.join(base, f"library-{shard_counter:06d}.tar")  # Use the pattern to make a shard_name
        if os.path.exists(shard_name):
            print(f"Shard {shard_name} already exists, skipping...")
        else:
            make_shard(shard_name, page_chunk)
        shard_counter += 1
    
write_dataset(args.data, args.shards, nsamples=args.maxcount)
