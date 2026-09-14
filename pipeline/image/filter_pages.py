import os
import glob
import json
from PIL import Image, ImageDraw
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Filter blank pages and pages consisting of only text from books.")
parser.add_argument("--books_dir", type=str, default="library/", help="Directory to the books images.")
parser.add_argument("--annotations_dir", type=str, default="library-jsons/", help="Directory to the JSON files.")
parser.add_argument("--filtered_dir", type=str, default="filtered-library/", help="Directory to store the filtered pages.")
parser.add_argument("--text_area_threshold", type=float, default=0.3,
                    help="Maximum fraction of the page covered by detected text.")
parser.add_argument("--blank_threshold", type=float, default=0.2,
                    help="Minimum edge-content score before text removal.")
parser.add_argument("--redacted_blank_threshold", type=float, default=0.3,
                    help="Minimum edge-content score after detected text is masked.")
args = parser.parse_args()

book_folders = glob.glob(args.books_dir + "/*")
book_paths = [bookpath for bookpath in book_folders if os.path.isdir(bookpath)] 
book_names = [os.path.basename(bookpath) for bookpath in book_paths] 

# Load the annotations
json_files = [json_file for json_file in glob.glob(args.annotations_dir + "/*.json") if os.path.splitext(json_file)[0].split("/")[-1] in book_names]
annotations = {
    os.path.splitext(json_file)[0].split("/")[-1]: json.load(open(json_file))
    for json_file in json_files
}

def is_blank(pil_image, threshold=0.2):
    # Convert PIL Image to OpenCV format
    image = np.array(pil_image.convert('L'))
    if image is None:
        print("Cannot convert image to an array, skipping.")
        return True, -1
    edges = cv2.Canny(image, 100, 200)
    if edges is None:
        print("Cannot compute image edges, skipping.")
        return True, -1

    # Check if there are any edges detected
    if edges.max() != 0:
        edges = edges / edges.max() # Normalize the edge data
    else:
        # If no edges are detected, the image is likely a blank page
        return True, 0
    stddev = np.std(edges)
    return stddev < threshold, stddev

def calculate_text_area(bounding_box):
    return (bounding_box[1][0] - bounding_box[0][0]) * (bounding_box[2][1] - bounding_box[0][1])

def calculate_mean_color(image_np, bounding_box, default_color):
    region_coords = (int(bounding_box[2][0]), int(bounding_box[0][1]), int(bounding_box[2][0]) + 1, int(bounding_box[2][1]))
    region = image_np[region_coords[1]:region_coords[3], region_coords[0]:region_coords[2]]
    if region.size > 0 and not np.isnan(region).any():
        if len(region.shape) == 3:
            return tuple(int(round(value)) for value in region.mean(axis=(0, 1)))
        else:
            return int(region.mean())
    else:
        return default_color

def draw_rectangle(draw, bounding_box, mean_color, padding = 12, width = 5):
    draw.rectangle(
        [
            (min(bounding_box[0][0], bounding_box[2][0]), min(bounding_box[0][1], bounding_box[2][1]) - padding),
            (max(bounding_box[0][0], bounding_box[2][0]), max(bounding_box[0][1], bounding_box[2][1]) + padding)
        ],
        fill=mean_color,
        width=width,
    )

def filter_books(text_area_ratio_threshold=0.3):
    """
    Filters books based on the text area ratio and Canny edge detection.

    This function iterates over each book and its pages. For each page, it calculates the ratio of the text area to the total area.
    If this ratio is above the provided threshold, the page is considered to have too much text and is not copied.
    Additionally, the function uses Canny edge detection to determine if a page is blank. If a page is blank, it is also not copied.
    The function does not return anything.

    Args:
        text_area_ratio_threshold (float, optional): The maximum ratio of text area to total area for a page to be considered. Defaults to 0.3.
    """

    filtered_count = 0
    for book_path in tqdm(book_paths, desc="Processing books"):
        book_name = os.path.basename(book_path)
        page_paths = glob.glob(book_path + "/*.jpg") # Get a list of all page paths in the book
        for pagepath in page_paths:  
            page_number = os.path.basename(pagepath)
            try:
                image = Image.open(pagepath)
                image_np = np.array(image, dtype=np.uint8)
                try:
                    h, w = image_np.shape[:2]
                except ValueError:
                    print(f"Unexpected image shape {image_np.shape} for image {pagepath}, skipping.")
                    continue
                page_area = h * w
                text_area = sum(calculate_text_area(obj["bounding_box"]) for obj in annotations.get(book_name, []) if obj["page_number"] == page_number)
                text_area_ratio = text_area / page_area
            except Exception as e:
                print(f"Cannot identify image file {pagepath}, skipping due to error: {e}")
                continue
            
            is_blank_page, stddev = is_blank(image, threshold=args.blank_threshold)
            if stddev == -1:
                continue
            if text_area_ratio < text_area_ratio_threshold and not is_blank_page:
                # Redact and check if the page is blank again
                draw = ImageDraw.Draw(image)
                default_color = tuple(image_np.mean(axis=(0, 1)).astype(int)) if len(image_np.shape) == 3 else int(image_np.mean())                
                for obj in annotations.get(book_name, []):
                    if obj["page_number"] == page_number:
                        mean_color = calculate_mean_color(image_np, obj["bounding_box"], default_color)
                        draw_rectangle(draw, obj["bounding_box"], mean_color)
                is_blank_page, stddev = is_blank(image, threshold=args.redacted_blank_threshold)
                if stddev == -1:
                    continue
                if not is_blank_page:
                    filtered_count += 1
                    filtered_file_name = f"{book_name}_{page_number.replace('.jpg', '')}.jpg"
                    filtered_path = os.path.join(args.filtered_dir, filtered_file_name)
                    os.makedirs(os.path.dirname(filtered_path), exist_ok=True) 
                    shutil.copy(pagepath, filtered_path) 
    print("The number of books that are in the threshold: ", filtered_count)

filter_books(args.text_area_threshold)
