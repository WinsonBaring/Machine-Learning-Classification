# Machine Learning Classification

## Overview
This repository contains code for a machine learning classification model. The model is trained to classify images into different categories using deep learning techniques.

## Classification Model

### How to Run
1. Create a Virtual Environment: 
python -m venv venv

2. Run Virtual Environment: 
./venv/Scripts/activate

3. Install Requirements: 
pip install -r requirement.txt

4. Run Streamlit: 
In the terminal, execute the following command:
streamlit run app.py


### How to Train: Multiclass Classification
- classification_model.h5: This file contains the trained classification model.
- You can load this model using `model.load_model("classification_model.h5")`.

### Notes on Model Loading
- Model Compilation: After loading the model, you can compile it with different configurations such as weights and activations.
- Model Training: You can further train the loaded model and apply additional callbacks as needed.

### Where to Download Dataset
1. Download Dataset: Download the dataset from the provided link: [Dataset Link](https://drive.google.com/drive/folders/1nLwmRSAGseFITBUGG83JikS4k5hOMb_4?usp=drive_link).
2. Data Directory: Create a data directory inside the `src` folder.
3. Place Dataset: Place the downloaded dataset in the data directory.

## Additional Resources
- Website Link: [Access additional resources related to the project.](https://machine-learning-classification.streamlit.app/)

Feel free to explore and contribute to this project!
