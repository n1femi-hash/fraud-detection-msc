# Fraud Detection with Machine Learning

This project implements a machine learning pipeline to detect fraudulent financial transactions using Python. It leverages a dataset obtained from Kaggle and organizes the development into modular folders for clarity and maintainability.

## 💻 System Information

- **OS**: Windows 11 Home  
- **Device**: Dell G15 5515  
- **GPU**: NVIDIA RTX 3050 4GB VRAM 
- **RAM**: 16GB 4800MHz
- **Storage**: 1TB SSD  
- **Python Version**: 3.9  
- **CUDA Version**: 12.7
- **NVIDIA Driver**: 566.41

## 📁 Project Structure

```
project-root/
│
├── assets/           # Contains datasets (fraudTrain.csv, fraudTest.csv)
│   └── fraudTrain.csv
│   └── fraudTest.csv
│
├── misc/             # Contains auxiliary files like job sector mapping
│   └── job_sectors.json
│
├── src/              # Source code for training, preprocessing, evaluation, etc.
│   └── <your_python_files>.py
│
├── requirements.txt  # Python dependencies
└── README.md
```

## 📦 Setup Instructions

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
    ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt --no-cache-dir
   ```

4. Make sure you download the dataset from Kaggle:

   * URL: [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
   * Save `fraudTrain.csv` and `fraudTest.csv` inside the `assets/` folder.


---

### 🧹 `data_cleaning.py` — Dataset Preprocessing

This script prepares the raw Kaggle dataset for machine learning by applying feature engineering, categorizing continuous variables, and reducing dimensionality. It produces a cleaned and enriched dataset optimized for fraud classification.

#### 🧰 Key Functionalities

1. **File Loading**

   * Loads `fraudTrain.csv` and `fraudTest.csv` from the `assets/` directory.
   * Loads auxiliary mappings from:

     * `job_sector.json` — for job-to-sector mapping
     * `age_group.json` — for custom age group classification

2. **Feature Engineering**

   * `extract_age`: Calculates age from the `dob` column.
   * `group_age`: Buckets ages into named age groups (e.g., Youth, Adult, Senior).
   * `group_amt`: Categorizes transaction amount magnitude (units, tens, hundreds+).
   * `set_job_sector`: Maps job titles to predefined sectors.
   * `extract_date`: Extracts transaction month and hour from the timestamp.
   * `get_time_of_day`: Converts transaction hour to time-of-day buckets (Morning, Evening, etc.).
   * `get_season`: Classifies month into seasonal categories (Winter, Spring, etc.).
   * `calculate_distance`: Computes geospatial distance between user and merchant via Haversine formula.
   * `group_distance`: Assigns distance into one of three buckets: `Nearest`, `Near`, `Far`.
   * `group_population`: Groups `city_pop` into three tiers: `Few`, `Average`, `Populous`.

3. **Column Cleanup**

   * Drops redundant, high-cardinality, or irrelevant fields such as:

     * `cc_num`, `dob`, `trans_num`, `street`, `city`, `merchant`, `lat`, `long`, `zip`, etc.

4. **Final Output**

   * Saves cleaned datasets as:

     * `assets/cleaned_fraud_train.csv`
     * `assets/cleaned_fraud_test.csv`

---

#### 🗂️ Output Schema After Cleaning

| Feature                | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| `category`             | Transaction category (e.g., gas\_transport, grocery\_net) |
| `gender`               | Gender of the card owner                                  |
| `amt_group`            | Transaction amount grouped by magnitude                   |
| `age_group`            | User age grouped into categories                          |
| `job_sector`           | Sector derived from job title                             |
| `trans_month`          | Month of the transaction                                  |
| `trans_hour`           | Hour of the transaction                                   |
| `time_of_day`          | Grouped time-of-day (Morning, Evening, etc.)              |
| `season`               | Seasonal grouping of month (Winter, Spring, etc.)         |
| `city_pop_group`       | City population tier                                      |
| `trans_distance_group` | Geographical distance bucket between user and merchant    |
| `is_fraud`             | Label column (1 = fraud, 0 = non-fraud)                   |

---

#### 💡 Usage

1. Ensure the following files are available:

   * `assets/fraudTrain.csv`
   * `assets/fraudTest.csv`
   * `misc/job_sector.json`
   * `misc/age_group.json`

2. Run the script with:

```bash
python src/data_cleaning.py
```

This will generate cleaned and feature-rich datasets in the `assets/` folder, ready for augmentation or modeling.

---

### 🧼 `data_preprocessing.py` — Balancing & One-Hot Encoding

This script performs **two key preprocessing steps** to prepare the dataset for machine learning models:

1. **Class Balancing via Downsampling**
2. **Vectorization via One-Hot Encoding**

It operates on the cleaned datasets and generates a final, fully encoded and class-balanced output for training and testing.

---

#### 🧠 Why This Step Matters

* The dataset is **highly imbalanced**, with far more non-fraud than fraud transactions.
* Many columns are **categorical**, requiring transformation into numeric vectors via **one-hot encoding**.
* Machine learning models like logistic regression, decision trees, and neural networks require **numerically encoded** and **balanced** input.

---

#### ⚙️ What the Script Does

1. **Load Datasets**
   Reads the cleaned datasets:

   * `cleaned_fraud_train.csv`
   * `cleaned_fraud_test.csv`

2. **Display Before Processing**
   Prints class distribution (`is_fraud`) and previews first few rows of both datasets.

3. **Class Balancing via Downsampling**

   * Finds the smallest class count (`is_fraud == 1` or `0`)
   * Samples that same number of rows from both classes
   * Ensures the resulting dataset has a **50:50** fraud-to-non-fraud ratio

4. **Vectorization via One-Hot Encoding**

   * Applies `pd.get_dummies()` to **every column except `is_fraud`**
   * Avoids feature name conflicts using prefixes
   * Concatenates one-hot vectors into the dataframe

5. **Column Alignment**

   * Ensures both train and test sets have the **same columns** (even if some categories exist in only one set)
   * Fills missing columns with `0.0` as default

6. **Display After Processing**

   * Prints balanced class counts
   * Previews the one-hot encoded data

7. **Final Output**

   * Saves final preprocessed files as:

     * `preprocessed_fraud_train.csv`
     * `preprocessed_fraud_test.csv`

---

#### 📦 Output File Schema

| Column Pattern                | Description                                     |
| ----------------------------- | ----------------------------------------------- |
| `category_gas_transport`      | One-hot vector from original categorical fields |
| `job_sector_Information Tech` | Sector one-hot encoded                          |
| `time_of_day_Morning`         | Time of day encoding                            |
| `season_Summer`               | Seasonal encoding                               |
| `city_pop_group_Populous`     | Population group encoding                       |
| `trans_distance_group_Far`    | Distance group encoding                         |
| ...                           | ... (hundreds of features from encoded columns) |
| `is_fraud`                    | Target label (0 = legit, 1 = fraud)             |

---

#### 💡 Usage

Run the script with:

```bash
python src/data_preprocessing.py
```

Make sure the cleaned datasets exist in the following paths:

```
assets/cleaned_fraud_train.csv
assets/cleaned_fraud_test.csv
```

This will generate:

```
assets/preprocessed_fraud_train.csv
assets/preprocessed_fraud_test.csv
```

These final datasets are ready to be passed into your training pipeline.

---

### 🧠 `model_training.py` — CNN-Based Fraud Classifier

This script builds and trains a **1D Convolutional Neural Network (CNN)** to detect fraudulent transactions from the preprocessed data. It uses TensorFlow/Keras for deep learning and evaluates performance using key classification metrics.

---

#### 📦 Pre-requisites

* Input files:

  * `preprocessed_fraud_train.csv`
  * `preprocessed_fraud_test.csv`
* These must exist in the `assets/` directory before running this script.

---

#### 🧰 What the Script Does

1. **Load Preprocessed Data**

   * Reads fully encoded and balanced CSV files into memory.
   * Shuffles the training data for randomness and improved generalization.

2. **Prepare Features and Labels**

   * Separates the `is_fraud` column as the label.
   * Drops `is_fraud` from feature vectors for both training and testing datasets.

3. **Define CNN Model Architecture**

   * **Conv1D Layer**: Extracts temporal patterns across input features.
   * **MaxPooling1D**: Reduces dimensionality after convolution.
   * **Flatten**: Converts 2D feature map to 1D for Dense layers.
   * **BatchNormalization**: Stabilizes and accelerates training.
   * **Dense Layers**: Learn deep representations (512 and 256 units).
   * **Dropout Layers**: Regularize and prevent overfitting (30% rate).
   * **Output Layer**: Single sigmoid neuron for binary classification.

4. **Compile the Model**

   * Optimizer: `Adam`
   * Loss: `binary_crossentropy`
   * Metrics:

     * `accuracy`
     * `Precision`
     * `Recall`
     * `AUC`

5. **Train the Model**

   * `epochs`: 15
   * `batch_size`: 128
   * `validation_split`: 40% of the training data used for validation

6. **Evaluate the Model**

   * Tests on unseen data and prints all metrics

7. **Print Model Summary**

   * Displays a layer-by-layer breakdown and parameter count

---

#### 📊 Output

Sample console output during training and evaluation:

```
Model Training
======================
Epoch 1/15
...
Epoch 15/15
...
======================
Model Evaluation
======================
Test loss: 0.19 - Accuracy: 0.92 - Precision: 0.91 - Recall: 0.90 - AUC: 0.96
======================
Model Summary
======================
Model: "sequential"
...
Total params: 180,097
Trainable params: 179,841
```

---

#### 🚀 Run the Model

```bash
python src/model_training.py
```

---

### ⚠️ Note on Data Augmentation

Data augmentation was intentionally excluded from this project.

Initial attempts at synthetic data generation—whether by sampling from feature distributions or using techniques like SMOTE—consistently introduced excessive **noise** and **overfitting**, especially due to the presence of high-cardinality and categorical fields such as `job_sector`, `merchant`, and `category`.

Rather than risk reduced generalization and misleading performance metrics, the pipeline relies on real, cleaned data combined with class balancing via **downsampling**, resulting in a more stable and trustworthy training process.
