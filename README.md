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

This script prepares the raw Kaggle dataset for machine learning by extracting relevant features, computing distances, and encoding job information into sectors.

#### 🧰 Key Functionalities

1. **File Loading**

   * Loads the training and test CSVs from the `assets/` directory.
   * Loads job-sector mappings from a JSON file in the `misc/` directory.

2. **Feature Engineering**

   * **`extract_age`**: Converts `dob` to an integer age (in years).
   * **`extract_date`**: Extracts transaction month and hour from the transaction timestamp.
   * **`calculate_distance`**: Uses the Haversine formula to compute distance (in miles) between user and merchant.
   * **`set_job_sector`**: Maps job titles to job sectors using a lookup dictionary, defaulting to `"Other"` if no match is found.

3. **Column Cleanup**

   * Drops irrelevant or high-cardinality features: `cc_num`, `unix_time`, `dob`, `lat`, `long`, etc.

4. **Final Output**

   * Saves the cleaned training and test datasets as:

     * `assets/cleaned_fraud_train.csv`
     * `assets/cleaned_fraud_test.csv`

#### 🗂️ Output Schema After Cleaning

| Feature                | Description                                                                          |
|------------------------|--------------------------------------------------------------------------------------|
| `category`             | Transaction type/category                                                            |
| `amt`                  | Transaction amount                                                                   |
| `gender`               | Gender of card owner                                                                 |
| `city`                 | Geographical data                                                                    |
| `city_pop_group`       | Grouped city population                                                              |
| `merchant`             | Zip and merchant info                                                                |
| `is_fraud`             | Label: 1 for fraud, 0 otherwise                                                      |
| `age_group`            | Age group; a secondary derivative of age from DOB                                    |
| `job_sector`           | Grouped sector derived from job title                                                |
| `trans_month`          | Month of transaction extracted from timestamp                                        |
| `trans_hour`           | Hour of transaction                                                                  |
| `trans_distance_group` | Haversine distance between user and merchant locations categorised based on nearness |

#### 💡 Usage

Place this script in the `src/` folder and run:

```bash
python src/clean_data.py
```

Ensure the original data (`fraudTrain.csv`, `fraudTest.csv`) and `job_sector.json` are already placed in their respective folders (`assets/` and `misc/`).

---

### 🔁 `data_augmentation.py` — Synthetic Data Generation

This script increases the size of your training and test datasets by generating realistic synthetic samples. It uses probability-based sampling to maintain the statistical distribution of original features, thereby expanding the dataset without introducing noise.

#### 🧠 Why Augmentation?

The original fraud detection dataset is highly imbalanced and may be insufficient for robust model training. Augmentation helps:

* Boost sample size by a factor of 5
* Maintain original feature distributions
* Improve generalization of machine learning models

#### ⚙️ How It Works

1. **Load Cleaned Data**

   * Reads cleaned training and testing CSV files from the `assets/` directory.

2. **Value Distribution Extraction**

   * For each column, calculates normalized value counts (frequencies) using:

     ```python
     df[col].value_counts(normalize=True)
     ```

3. **Synthetic Row Generation**

   * Each new row is generated by randomly selecting values from the original distributions using `numpy.random.choice` with weighted probabilities.

4. **Dataset Expansion**

   * Generates 4× more rows than the original dataset.
   * Concatenates synthetic rows with original data.
   * Drops any accidental duplicates for uniqueness.

#### 🧪 Example

If the original training dataset has `30,000` rows, this script will generate `270,000` synthetic rows, resulting in `~300,000` rows after augmentation.

#### 💾 Output

The script augments the datasets **in memory**, ready to be used for further training or saved manually if needed:

```python
dataframe_cleaned = augment_data(dataframe_cleaned)
dataframe_test_cleaned = augment_data(dataframe_test_cleaned)
```

#### 🧬 Notes

* The augmentation is purely **statistical** — it doesn't apply domain-specific transformations.
* This method **preserves class imbalance** unless additional sampling logic is introduced.

#### 🚀 Usage

Run the script with:

```bash
python src/data_augmentation.py
```

Ensure that the cleaned datasets `cleaned_fraud_train.csv` and `cleaned_fraud_test.csv` are already available in the `assets/` directory.

---

Great! Here's a clear and structured Markdown section you can add to your `README.md` (or internal project documentation) to describe the contents and purpose of your `data_preprocessing.py` file:

---

### 🧼 `data_preprocessing.py` — Dataset Balancing via Downsampling

After augmentation, this script applies **downsampling** to ensure class balance in both the training and test datasets. It specifically addresses the class imbalance between fraudulent and non-fraudulent transactions.

#### 🧠 Why Downsampling?

Fraud datasets typically have a **severe class imbalance** — most transactions are non-fraudulent. This can bias the model toward always predicting the majority class. Downsampling ensures that both classes have equal representation by reducing the majority class to match the count of the minority class.

#### ⚙️ What the Script Does

1. **Load Augmented Datasets**

   * Reads `augmented_fraud_train.csv` and `augmented_fraud_test.csv` from the `assets/` directory.

2. **Apply Downsampling**

   * Calculates the **minimum class count** between fraud and non-fraud.
   * Applies `.sample()` on each class group to retain an equal number of samples from both classes.
   * Resets the index to produce a clean, balanced dataset.

#### 📦 Function Breakdown

```python
def data_downsampling(dataframe):
    minimum_count = dataframe['is_fraud'].value_counts().min()
    grouped_dataframe = dataframe.groupby('is_fraud')
    grouped_dataframe = grouped_dataframe.apply(lambda x: x.sample(minimum_count))
    dataframe = grouped_dataframe.reset_index(drop=True)
    return dataframe
```

This function is applied to both training and test datasets:

```python
dataframe_augmented = data_downsampling(dataframe_augmented)
dataframe_test_augmented = data_downsampling(dataframe_test_augmented)
```

#### 💾 Output

* Balanced datasets held in memory as `dataframe_augmented` and `dataframe_test_augmented`.
* Ready for feature scaling, encoding, or model training.

---

### ⚠️ Note on Augmentation Challenges

* **SMOTE Not Applicable**:
  Traditional oversampling techniques like **SMOTE** were not suitable for this dataset, as not all features are numeric. Categorical columns such as `job_sector`, `merchant`, and `category` make SMOTE incompatible without additional preprocessing or encoding.