# Turkish Sentiment Analysis

This project focuses on Turkish sentiment analysis using natural language processing and machine learning techniques. The aim is to classify Turkish texts into sentiment categories such as **Negative**, **Notr**, and **Positive**.

## Project Overview

Sentiment analysis is a common natural language processing task that aims to identify the emotional tone or opinion expressed in a text. In this project, Turkish text data is processed and classified using a traditional machine learning pipeline.

The project includes:

- Turkish text preprocessing
- TF-IDF feature extraction
- Logistic Regression classification
- Model evaluation using accuracy, precision, recall, and F1-score
- Sentiment prediction on new Turkish texts

## Technologies Used

- Python
- pandas
- scikit-learn
- TF-IDF
- Logistic Regression
- VS Code
- Git & GitHub

## Dataset

The dataset includes three columns:

| Column | Description |
|---|---|
| `text` | Turkish text data |
| `label` | Sentiment label |
| `dataset` | Dataset source information |

The dataset contains separate train and test files:

```text
data/
├── train.csv
└── test.csv
```

The train and test files are not included in this repository because of file size and dataset usage limitations. Users should manually place the dataset files inside the `data/` folder before running the project.

The dataset labels are:

- `Negative`
- `Notr`
- `Positive`

Note: The dataset uses the label `Notr` for neutral sentiment.

## Project Structure

```text
turkish-sentiment-analysis/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│
├── reports/
│   └── classification_report.txt
│
├── src/
│   ├── check_data.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   └── turkish_sentiment_model.pkl
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Methodology

The project follows a basic machine learning workflow:

1. Load the train and test datasets
2. Clean the Turkish text data
3. Convert text into numerical features using TF-IDF
4. Train a Logistic Regression model
5. Evaluate the model on the test set
6. Save the trained model
7. Predict sentiment for new Turkish texts

## Text Preprocessing

The preprocessing step includes:

- Converting text to lowercase
- Removing URLs
- Removing numbers and unnecessary symbols
- Preserving Turkish characters such as `ç`, `ğ`, `ı`, `ö`, `ş`, and `ü`
- Removing extra spaces

## Model

The baseline model uses a machine learning pipeline consisting of:

- `TfidfVectorizer`
- `LogisticRegression`

The TF-IDF vectorizer converts text into numerical features, while Logistic Regression performs the sentiment classification task.

## Results

The baseline Logistic Regression model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 0.9265 |
| Macro F1-score | 0.89 |
| Weighted F1-score | 0.93 |

The model was evaluated on **48,965** test samples.

## Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.70 | 0.88 | 0.78 | 5,656 |
| Notr | 0.95 | 0.98 | 0.97 | 17,092 |
| Positive | 0.97 | 0.90 | 0.93 | 26,217 |

## Result Interpretation

The model achieved strong overall performance with an accuracy of **92.65%**. It performed especially well on the **Notr** and **Positive** classes.

However, the **Negative** class had lower precision compared to the other classes. This suggests that some neutral or positive texts may have been incorrectly classified as negative. Therefore, further improvements may be needed to improve class-level balance and reduce misclassification.

## Confusion Matrix

```text
[[ 4977   142   537]
 [  153 16803   136]
 [ 1956   677 23584]]
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/barisakk00/turkish-sentiment-analysis.git
```

```bash
cd turkish-sentiment-analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

For Windows:

```bash
.venv\Scripts\activate
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the dataset

Place the train and test files inside the `data/` folder:

```text
data/train.csv
data/test.csv
```

### 6. Train the model

```bash
python src/train_model.py
```

After training, the model will be saved under:

```text
models/turkish_sentiment_model.pkl
```

The classification report will be saved under:

```text
reports/classification_report.txt
```

### 7. Run prediction

```bash
python src/predict.py
```

## Example Predictions

Example input texts:

```text
Bu ürün gerçekten çok güzel, çok memnun kaldım.
Hiç beğenmedim, çok kötü bir deneyimdi.
Film fena değildi ama daha iyi olabilirdi.
Kargo zamanında geldi.
```

The model predicts the sentiment label for each input text.

## Limitations

This is a baseline machine learning project. Although the overall accuracy is high, the model has some limitations:

- It uses traditional TF-IDF features rather than contextual word embeddings.
- It may not fully capture sarcasm, irony, or context-dependent meanings.
- The negative class has lower precision than the other classes.
- The model performance may vary on texts from different domains.

## Future Improvements

Planned improvements include:

- Comparing multiple machine learning models
- Adding Naive Bayes and Linear SVM classifiers
- Improving Turkish text preprocessing
- Adding Turkish stopword removal
- Performing hyperparameter tuning
- Deploying the model using Streamlit or Hugging Face Spaces
- Fine-tuning a Turkish BERT model for comparison

## Author

Barış Ak  
Artificial Intelligence and Data Engineering Student  
Istanbul University