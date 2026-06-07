import os
import glob
from tqdm import tqdm
from extract_embeddings import FeatureExtractor
from database import VectorDatabase
import config

def main():
    print("Initializing PyTorch Model and FAISS Database...")
    extractor = FeatureExtractor()
    db = VectorDatabase()

    # Find all images in the data directory
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG'):
        image_paths.extend(glob.glob(os.path.join(config.DATA_DIR, ext)))

    if not image_paths:
        print(f"No images found in {config.DATA_DIR} folder!")
        return

    print(f"Found {len(image_paths)} images. Starting extraction...")
    
    vectors = []
    valid_paths = []

    for path in tqdm(image_paths):
        feature = extractor.extract(path)
        if feature is not None:
            vectors.append(feature)
            valid_paths.append(path)

    print("Adding vectors to database...")
    db.add_vectors(vectors, valid_paths)
    db.save()
    print("Indexing complete! You are ready to run the app.")

if __name__ == "__main__":
    main()
