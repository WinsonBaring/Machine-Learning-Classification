
import os
import cv2
import numpy as np
def load_images_and_labels(data_dir,label):
    images = []
    labels = []

    # Iterate through files in the data directory
    for filename in os.listdir(data_dir):
        # Read the image
        image_path = os.path.join(data_dir, filename)
        image = cv2.imread(image_path)
        print(image_path)
        if image is not None:
            # Preprocess the image (resize, normalize, etc.)
            image = cv2.resize(image, (100, 100))
            image = image / 255.0  # Normalize pixel values
            # image = np.expand_dims(image, axis=0)
            images.append(image)
            # Add label (assuming all images are cheetah)
            labels.append(label)  # One-hot encoding for "cheetah" label
      # Convert labels to one-hot encoding
    return np.array(images), np.array(labels)

# Path to the directory containing the images
# cheetah_label = [0, 0]
# hyena_label = [1, 0]
# jaguar_label = [0, 1]
# tiger_label = [1, 1]


hyena_label= [1, 0, 0]
cheetah_label= [0, 1, 0]
jaguar_label = [1, 1, 0]
tiger_label = [0, 0, 1]

