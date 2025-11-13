# --- Task 3: SVM Cat & Dog Trainer ---
# This is the "trainer" script.
# Its only job is to load all my images from the 'data_subset' folder,
# learn what a cat and a dog looks like based on their HOG features,
# and then save that "knowledge" (the trained model) into a file.
# This will take a while, but I only have to run it ONCE.

import os
import cv2  # OpenCV for image processing
from skimage.feature import hog  # This is the HOG feature extractor
import numpy as np
from sklearn.svm import SVC  # Support Vector Machine (Classifier)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  # This is for saving my trained model
import warnings

# Scikit-learn can be noisy, so I'll ignore some warnings
warnings.filterwarnings('ignore')

# --- 1. Setup ---

# I'm resizing all images to be the same size. 64x64 is a good, fast standard.
IMG_SIZE = (64, 64)

# My data subset folders
CAT_DIR = os.path.join('data_subset', 'cats')
DOG_DIR = os.path.join('data_subset', 'dogs')

# Where I'll save my trained model
MODEL_FILE = 'svm_model.joblib'

# My labels: 0 for cat, 1 for dog
LABELS = {'cats': 0, 'dogs': 1}

# --- 2. Load and Process Images ---
print(f"Starting image processing... This will take a few minutes.")

data = []  # This will hold all my HOG features (the 'X')
labels = []  # This will hold 0 (cat) or 1 (dog) (the 'y')

# Loop over my two directories: 'cats' and 'dogs'
for label_name, directory in [('cats', CAT_DIR), ('dogs', DOG_DIR)]:
    print(f"Processing '{label_name}' images...")

    # Get the label (0 or 1)
    label = LABELS[label_name]

    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        continue

    # Loop through every image file in that directory
    for i, filename in enumerate(os.listdir(directory)):
        # Full path to the image
        img_path = os.path.join(directory, filename)

        # Load the image using OpenCV
        img = cv2.imread(img_path)

        # Check if image loaded correctly
        if img is None:
            print(f"  Skipping {filename} (couldn't be read).")
            continue

        # My 3-step process for each image:
        # 1. Convert to Grayscale (HOG works on shape/shadows, not color)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Resize to my standard 64x64 size
        resized_img = cv2.resize(gray_img, IMG_SIZE)

        # 3. Calculate HOG features
        # HOG stands for Histogram of Oriented Gradients.
        # It's a "feature descriptor" that turns shape info into a long list of numbers
        # that my SVM can understand.
        hog_features = hog(resized_img,
                           orientations=9,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           transform_sqrt=True,
                           block_norm='L2-Hys')

        # Add the features and the label to my lists
        data.append(hog_features)
        labels.append(label)

        # A simple progress print so I know it's not crashed
        if (i + 1) % 100 == 0:
            print(f"  ...processed {i+1} {label_name} images.")

print(f"\nImage processing complete! Processed {len(data)} total images.")

# --- 3. Prepare for Training ---
# Convert my lists into NumPy arrays (scikit-learn loves these)
X = np.array(data)
y = np.array(labels)

# Check the shapes to make sure everything looks right
print(f"Data array shape: {X.shape}")  # Should be (2000, some_number)
print(f"Labels array shape: {y.shape}")  # Should be (2000,)

# Split my data: 80% for training, 20% for testing
# I'll use stratify=y to make sure my training and testing sets
# have the same percentage of cats and dogs.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

print("Data split into 80% training and 20% testing sets.")

# --- 4. Train the SVM Model ---
print("Training the Support Vector Machine (SVM)... This is the *real* wait.")

# Create the SVM model. I'll use a 'linear' kernel.
# It's fast and a good starting point.
# C=1.0 is a standard "regularization" parameter.
# probability=True is a bit slower, but it lets me get a "confidence score"
# which I might want to use in my app later.
model = SVC(kernel='linear', C=1.0, random_state=42, probability=True)

# Let's... GO!
model.fit(X_train, y_train)

print("Training complete!")

# --- 5. Evaluate the Model ---
print("Let's see how well it learned...")

# Make predictions on the 20% of data it's never seen
y_pred = model.predict(X_test)

# Check the accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

# This report is super useful. It shows 'precision' and 'recall'.
# 0 = cats, 1 = dogs
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['cats', 'dogs']))

# --- 6. Save the Model ---
# I'm done! Now I save my trained 'model' to a file.
# My app.py script will just load this file,
# so I never have to re-train this again.
joblib.dump(model, MODEL_FILE)

print(f"\nModel saved successfully to '{MODEL_FILE}'")
print("\n--- Task 3 Trainer Script Finished ---")
print("You can now run the app: 'streamlit run app.py'")