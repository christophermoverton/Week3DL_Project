
###  README.md for Week 3 DL Project

#  Week 3 Deep Learning Project: CNN for Cancer Detection

##  Overview
This project applies Convolutional Neural Networks (CNNs) to detect metastatic cancer in digital pathology images. The dataset consists of small image patches extracted from whole-slide images, and the goal is to classify them as benign (0) or malignant (1).

The challenge is based on a Kaggle competition, leveraging the PCam dataset, which is a smaller, patch-based version of Camelyon16.

##  Project Highlights
- Built a CNN Model: Optimized for 96x96 images.
- Trained on Large Dataset: Using batch training to handle memory limitations.
- Hyperparameter Tuning: Used KerasTuner to optimize model performance.
- Final Model Performance:
  - Training Accuracy: ~98.3%
  - Validation Accuracy: ~87.7%
  - Improvement over Baseline Models
- Predictions on Unlabeled Data: Processed `.tif` images from the test directory in batches.

---

##  Installation & Setup
###  Clone the Repository
```bash
git clone https://github.com/christophermoverton/Week3DL_Project.git
cd Week3DL_Project

---

### Setting Up the Conda Environment

To ensure all dependencies are installed correctly, follow these steps:

###  Install Anaconda (if not already installed)
Download and install Anaconda from: [🔗 https://www.anaconda.com/download](https://www.anaconda.com/download)

###  Create the Environment from `environment.yml`
Navigate to the project directory and run:
```bash
conda env create -f environment.yml
```

###  Activate the New Environment
```bash
conda activate my_env  # Replace "my_env" with the actual environment name from `environment.yml`
```

###  Verify Installed Packages
To check if all dependencies were installed correctly:
```bash
conda list
```

###  Updating the Environment (if needed)
If any new dependencies are added, update the environment with:
```bash
conda env update --file environment.yml --prune
```

---

###  Optional: Export the Environment
If you make changes to the environment, export an updated version:
```bash
conda env export > environment.yml
```

---

##  Dataset Download
This project uses the Histopathologic Cancer Detection dataset from Kaggle.  
📥 Download the dataset from Kaggle:  
🔗 [Histopathologic Cancer Detection Dataset](https://www.kaggle.com/competitions/histopathologic-cancer-detection)

###  Install Kaggle CLI (if not installed)
```bash
pip install kaggle
```
If you haven't already, set up Kaggle API credentials:
1. Go to [Kaggle API Settings](https://www.kaggle.com/account).
2. Click "Create New API Token" (this downloads `kaggle.json`).
3. Move `kaggle.json` to your home directory:
   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

###  Download the Dataset
Run:
```bash
kaggle competitions download -c histopathologic-cancer-detection
```
Unzip the dataset:
```bash
unzip histopathologic-cancer-detection.zip -d dataset/
```

---

### Put the downloaded Kaggle Dataset data in the right folders
- Place train and test images inside:
  ```
  /train  (for training images)
  /test   (for unlabeled test images)
  ```
  
---

##  Model Training
### Run the CNN Training Script
```bash
python train.py
```
- This will:
  - Load the dataset
  - Train a CNN model
  - Apply hyperparameter tuning
  - Save the final model (`cancer_detection_cnn.h5`)

---

##  Making Predictions on Unlabeled Images
### Run Predictions on `.tif` Images
```bash
python predict.py
```
- This will:
  - Process all `.tif` images in `/test`
  - Perform batch predictions using the trained model
  - Save results to `test_predictions.csv`

---

##  Results & Observations
| Model | Max Training Accuracy | Max Validation Accuracy | Observations |
|-----------|----------------|----------------|----------------|
| Baseline (20K Sample) | 88.8% | 80.7% | Limited dataset led to overfitting |
| Full Dataset (Batched) | 93.4% | 91.1% | Best balance of performance & generalization |
| Hyperparameter Tuned | 98.3% | 87.7% | Overfitting increased slightly |

### Key Findings
✅ Batch training on the full dataset improved generalization  
✅ Hyperparameter tuning significantly improved accuracy  
✅ Final model achieves strong validation accuracy (87.7%)  

---

##  Project Structure
```
Week3DL_Project/
│── train.py              # Model training script
│── predict.py            # Batch prediction script
│── cancer_detection_cnn.h5  # Trained model
│── requirements.txt      # Dependencies
│── test_predictions.csv  # Predictions on test images
│── /train                # Training images (Not in Git)
│── /test                 # Unlabeled test images (Not in Git)
│── README.md             # Project Documentation
```

---

##  Next Steps
- Improve Generalization: Add data augmentation.
- Reduce Overfitting: Apply regularization techniques.
- Experiment with Transfer Learning: Try ResNet, VGG16, EfficientNet.

---

##  License
This project is open-source under the MIT License.

---

##  Acknowledgments
- Kaggle PCam Dataset 
- TensorFlow/Keras for Deep Learning 

---

##  Contribute
Feel free to fork this repo and suggest improvements via pull requests! 

```
git clone https://github.com/christophermoverton/Week3DL_Project.git
```
If you find this useful, give it a ⭐ on GitHub!

---

##  Contact
For any questions, reach out via:
- GitHub Issues: [Open an Issue](https://github.com/christophermoverton/Week3DL_Project/issues)
- Email: christopher.overton@colorado.edu

