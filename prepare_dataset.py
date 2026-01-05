#!/usr/bin/env python3

import os
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


INPUT_FACE_DIR = "face" #1 class
INPUT_NO_FACE_DIR = "no_face" #0 class
OUTPUT_DIR = "clean_data"  

TARGET_SIZE = (224, 224) # ResNet18 requires 224x224
MAX_IMAGES_PER_CLASS = None # None = use all but can be modified for a smaller set


def setup_directories():
    Path(OUTPUT_DIR, "data", "faces").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR, "data", "no_faces").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR, "splits").mkdir(parents=True, exist_ok=True)
    
    print("Created output directories")



def process_images(input_dir, output_dir, label, class_name):

    print(f"\nProcessing {class_name} images from {input_dir}...")
    
    #find files
    image_files = list(Path(input_dir).glob("*.jpg")) + \
                  list(Path(input_dir).glob("*.png")) + \
                  list(Path(input_dir).glob("*.jpeg"))
    
    print(f"Found {len(image_files)} images")
    
    # limit if given
    if MAX_IMAGES_PER_CLASS and len(image_files) > MAX_IMAGES_PER_CLASS:
        import random
        random.seed(42)
        image_files = random.sample(image_files, MAX_IMAGES_PER_CLASS)
        print(f"Limited to {MAX_IMAGES_PER_CLASS} for class balance")
    
    processed = []
    
    # process by resizing and convert to RGB
    for i, img_path in enumerate(tqdm(image_files, desc=f"  Validating & resizing")):
        try:
            with Image.open(img_path) as img:
                #rgb
                img = img.convert('RGB')
                
                #resize
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
                
                #tidy filenames
                if class_name == "face":
                    new_filename = f"face_{i:06d}.jpg"
                else:
                    new_filename = f"no_face_{i:06d}.jpg"
                
                output_path = Path(output_dir) / new_filename
                
    
                img.save(output_path, 'JPEG', quality=95)
                

                processed.append({
                    'filename': str(output_path),
                    'label': label,
                    'class': class_name
                })
                
        except Exception as e:
            #skip corrupted images
            print(f"Skipped {img_path.name}: {e}")
            continue
    
    print(f"processed {len(processed)} images")
    return processed



def balance_classes(faces, no_faces):
    print(f" Faces: {len(faces)}")
    print(f" No faces: {len(no_faces)}")
    
    # making class sizes ==
    min_size = min(len(faces), len(no_faces))
    
    # randomly sample from larger class 
    import random
    random.seed(42)
    
    if len(faces) > min_size:
        faces = random.sample(faces, min_size)
    if len(no_faces) > min_size:
        no_faces = random.sample(no_faces, min_size)
    
    print(f"balanced to {min_size} images per class")
    return faces, no_faces

#70/15/15 split
def create_splits(data):

    train_val, test = train_test_split(
        data,
        test_size=0.15,
        random_state=42,
        stratify=[d['label'] for d in data]  # Keep class balance
    )
    

    train, val = train_test_split(
        train_val,
        test_size=0.15/0.85,  # This gives us 15% of original
        random_state=42,
        stratify=[d['label'] for d in train_val]
    )
    

    print(f" Train: {len(train)} images ({len(train)/len(data)*100:.1f}%)")
    print(f" Val: {len(val)} images ({len(val)/len(data)*100:.1f}%)")
    print(f" Test:{len(test)} images ({len(test)/len(data)*100:.1f}%)")
    
    return {
        'train': train,
        'val': val,
        'test': test
    }

def save_csv_files(splits, output_dir): 
    for split_name, split_data in splits.items():
        df = pd.DataFrame(split_data)
        
        csv_path = Path(output_dir) / "splits" / f"{split_name}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"saved {csv_path}")


def print_summary(splits):
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    total = sum(len(split) for split in splits.values())
    total_faces = sum(sum(1 for d in split if d['label'] == 1) 
                     for split in splits.values())
    total_no_faces = total - total_faces
    
    print(f"Total images: {total}")
    print(f" Faces: {total_faces} ({total_faces/total*100:.1f}%)")
    print(f"No faces:{total_no_faces} ({total_no_faces/total*100:.1f}%)")
    print()
    
    for split_name, split_data in splits.items():
        faces = sum(1 for d in split_data if d['label'] == 1)
        no_faces = len(split_data) - faces
        print(f"{split_name:5s}: {len(split_data):5d} total "
              f"({faces:4d} faces, {no_faces:4d} no_faces)")
    
    print()
    print(f"Image size: 224x224 pixels")
    print(f"Format:JPEG")
    print(f"Output: {OUTPUT_DIR}/")



def main():


    if not Path(INPUT_FACE_DIR).exists():
        print(f"Error: {INPUT_FACE_DIR} not found!")
        return
    if not Path(INPUT_NO_FACE_DIR).exists():
        print(f"Error: {INPUT_NO_FACE_DIR} not found!")
        return
    
    setup_directories()
    faces = process_images(
        INPUT_FACE_DIR,
        Path(OUTPUT_DIR) / "data" / "faces",
        label=1,
        class_name="face"
    )
    
    no_faces = process_images(
        INPUT_NO_FACE_DIR,
        Path(OUTPUT_DIR) / "data" / "no_faces",
        label=0,
        class_name="no_face"
    )
    
    if not faces or not no_faces:
        print("Error: No valid images found!")
        return
    

    faces, no_faces = balance_classes(faces, no_faces)

    all_data = faces + no_faces
    splits = create_splits(all_data)
    
    save_csv_files(splits, OUTPUT_DIR)
 
    print_summary(splits)


if __name__ == "__main__":

    try:
        import pandas
        import sklearn
        from PIL import Image
        from tqdm import tqdm
    except ImportError as e:
        print("Missing required packages!")
        print("Run: pip install pandas scikit-learn pillow tqdm")
        exit(1)
    
    main()