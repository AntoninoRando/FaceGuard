# Image samples and test samples data paths
SAMPLES_PATH = "data/initial_samples"
TEST_PATH = "data/probes_for_test"

# Embeddings storage
EMBEDDINGS_FOLDER = "embeddings"
GALLERY_PATH = "embeddings/gallery.pkl"
ENCRYPTION_KEY_FILE = "embeddings/gallery_key.key"

# Identification system parameters
IDENTIFICATION_THRESHOLD = 0.3  # Distance threshold for open-set identification (cosine distance)

# Verification system parameters
VERIFICATION_THRESHOLD = 0.3  # Distance threshold for 1:1 verification (cosine distance)
# Anti-spoofing parameters
ANTISPOOFING_THRESHOLD = 0.90  # Deep Learning Real face score threshold (0.0 to 1.0)
ANTISPOOFING_ENABLE_BY_DEFAULT = True  # Enable anti-spoofing by default in GUI