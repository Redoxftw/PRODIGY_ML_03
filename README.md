### Prodigy InfoTech ML Internship: Task 3

This repository contains the completed Task 3 for the Prodigy InfoTech Machine Learning Internship.

Task: Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.

### 🚀 Live Demo

You can try the live classifier app I built with Streamlit. Upload your own photo of a cat or a dog to see how the model does!

🔗 Live App URL: https://vishwash-ml-task-03.streamlit.app/



### 🎯 Project Overview & Key Insights

This was a classic computer vision task with a twist. The goal was to use a "classical" machine learning model (SVM), not a modern Deep Learning one (like a CNN).

- The Challenge: An SVM cannot understand raw pixels. It needs a list of numbers (a "feature vector") that describes the image.

- The Solution: I used Histogram of Oriented Gradients (HOG). This is a feature extraction technique that analyzes the shape and outlines of an image and converts it into a vector.

- The Process:

1. I created a balanced subset of 2,000 images (1,000 cats, 1,000 dogs) from the main dataset.

2. I wrote ```task_03.py``` to loop through all 2,000 images, convert them to grayscale, resize them (to 64x64), and extract their HOG features.

3. I trained an SVM classifier on these HOG vectors.

4. I saved the final, trained model as ```svm_model.joblib``` so my app could load it instantly.

### 🧠 My "Knowledge Gained" (The "Bad" Predictions)

This was the best part of the task! My model gets a decent accuracy on its test data, but when I gave it new images... it failed hilariously.

- It classified a real photo of a cat as a DOG.

- It classified a clip-art drawing of a dog as a CAT.

This isn't a bug; it's the lesson. My model is only as smart as the data it was trained on.

1. Domain Mismatch: The clip-art dog looked nothing like the 1,000 real photos of dogs it was trained on, so it got confused.

2. Weak Features: HOG is only looking at shapes and outlines. The "shape" of that cat (pointy ears, snout) must have looked more like the "dog" HOG features it learned than the "cat" features.

3. This is why the world moved to Deep Learning (CNNs). A CNN learns to see "whiskers" and "fur," not just "outlines." This task was a perfect demonstration of the limitations of classical ML in computer vision.

### 📂 Files in this Repository

- ```task_03.py```: The "Trainer" script. You only run this once. It processes all 2,000 images, trains the SVM, and saves the ```svm_model.joblib``` file.

- ```app.py```: The "Inference" script. This is the interactive Streamlit web app. It just loads the ```svm_model.joblib``` file and uses it to classify any image you upload.

- ```svm_model.joblib```: The pre-trained "brain" of my model.

- ```requirements.txt```: All the Python libraries needed to run this project.

- ```packages.txt```: A special file to tell Streamlit Cloud to install the Linux libraries needed for ```opencv-python```.

- ```.python-version```: A file to tell Streamlit Cloud to use Python 3.11, which is stable for these libraries.

- ```.gitignore```: This is critical. It tells Git to ignore the ```venv/``` and ```data_subset/``` folders, so I don't upload 2,000 images to GitHub.

### 🏃 How to Run This Project

1. Clone the repository:
```
git clone https://github.com/Redoxftw/PRODIGY_ML_03.git
cd PRODIGY_ML_03
```

2. Create and activate a virtual environment:
(Using Python 3.11 is recommended)
```
py -3.11 -m venv venv
.\venv\Scripts\activate
```

3. Install the required libraries:
```
pip install -r requirements.txt
```

4. Run the interactive Streamlit app:
(You don't need to run task_03.py since the svm_model.joblib is already included!)
```
streamlit run app.py
```