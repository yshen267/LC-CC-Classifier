import csv
import os
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm
from skimage import io, color, util, exposure

# Feature extraction
def extract_features(image_path):
  # Read Image
  image = io.imread(image_path)

  # Extract LBP features
  greyscale_image = color.rgb2gray(image)
  lbp = local_binary_pattern(greyscale_image, 8, 1, method="uniform")

  # Normalize LBP histogram
  unique_values = np.unique(lbp.ravel())
  hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.append(unique_values, unique_values.max()+1))
  hist_lbp = hist_lbp.astype("float")
  hist_lbp /= (hist_lbp.sum() + 1e-7)

  # Extract LAB color features
  lab_image = color.rgb2lab(image)

  # Only use A and B channels
  a_channel = lab_image[:, :, 1]
  b_channel = lab_image[:, :, 2]
  hist_A, _ = exposure.histogram(a_channel)
  hist_B, _ = exposure.histogram(b_channel)

  # Normalize histograms
  hist_A = hist_A / hist_A.sum()
  hist_B = hist_B / hist_B.sum()

  feature_vector = np.hstack((hist_lbp.flatten(), hist_A.flatten(), hist_B.flatten()))

  return feature_vector

# Label vector
def extract_label(image_path):
  label_map = {'colon_aca' : 0, 'colon_n' : 1, 'lung_aca' : 2, 'lung_n' : 3, 'lung_scc' : 4}
  subfolder = os.path.basename(os.path.dirname(image_path))
  if subfolder in label_map:
    return label_map[subfolder]
  else:
    return -1
      
# Parallel Processing
def process_batch(batch, lock, csv_file, progress_counter):
    feature_list = []
    
    for image_path in batch:
        features = extract_features(image_path)
        label = extract_label(image_path)
        features_labeled = np.append(features, label)
        feature_list.append(features_labeled.tolist())

        with lock:
          progress_counter.value += 1

    with lock:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(feature_list)