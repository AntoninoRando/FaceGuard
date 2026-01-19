"""
Script to initialize the gallery with all face embeddings from initial samples.
This will process all images in data/initial_samples/ and create gallery.pkl.
"""

from sample_utils import initialize_gallery

if __name__ == "__main__":
    print("Starting gallery initialization...")
    print("This will process all images in data/initial_samples/")
    print("=" * 60)
    
    gallery = initialize_gallery()
    
    print("\nGallery initialization completed successfully!")
    print(f"Gallery contains {len(gallery['embeddings'])} embeddings")
    print(f"Across {len(set(gallery['labels']))} classes")
