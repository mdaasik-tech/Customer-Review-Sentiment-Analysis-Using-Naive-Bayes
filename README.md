# Customer Review Sentiment Analysis Using Naive Bayes

A machine learning project that classifies customer reviews into
**Positive**, **Negative**, or **Neutral** sentiment using the
**Multinomial Naive Bayes** algorithm.

**Author:** Mohammed Aasik.A

-   **LinkedIn:** [Mohammed
    Aasik](https://www.linkedin.com/in/mohammed-aasik-aspiring-machine-learning-engineer-257787433/)
-   **GitHub:** [mdaasik-tech](https://github.com/mdaasik-tech)

------------------------------------------------------------------------

## Project Overview

Customer reviews contain useful feedback about products and services.
This project uses text-processing and machine-learning techniques to
predict the sentiment expressed in a review.

The notebook explores the dataset, converts review text into numerical
features, trains a Naive Bayes classifier, evaluates its predictions,
and tests the model on new sample reviews.

## Objectives

-   Explore customer review data and inspect missing values.
-   Understand the distribution of sentiment classes.
-   Convert text into numerical features using `CountVectorizer`.
-   Train a `MultinomialNB` classification model.
-   Evaluate predictions using accuracy, a confusion matrix, and a
    classification report.
-   Predict sentiment for new customer reviews.

## Dataset

The supplied dataset contains **2,000 rows and 9 columns**.

  Column                  Description
  ----------------------- ----------------------------------------------
  `review_id`             Unique review identifier
  `product_category`      Product category
  `rating`                Customer rating
  `review_text`           Written customer review
  `review_length_chars`   Review length in characters
  `word_count`            Number of words in the review
  `helpful_votes`         Number of helpful votes
  `verified_purchase`     Whether the purchase was verified
  `sentiment`             Target label: Positive, Negative, or Neutral

The target column is `sentiment`. The notebook uses `review_text` as the
model input and drops `review_id`, which is an identifier rather than
meaningful review content.

### Sentiment distribution

  Sentiment     Number of reviews
  ----------- -------------------
  Positive                    888
  Negative                    611
  Neutral                     501

## Technologies Used

-   Python
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Jupyter Notebook / Google Colab

## Machine Learning Workflow

1.  **Load the dataset** using Pandas.
2.  **Inspect the data** with `head()`, `info()`, and missing-value
    checks.
3.  **Explore sentiment distribution** with a count plot.
4.  **Separate features and target**:
    -   Input (`X`): `review_text`
    -   Target (`y`): `sentiment`
5.  **Split the data** into training and testing sets using an 80/20
    split and `stratify=y`.
6.  **Vectorize the text** with `CountVectorizer`. Fit the vectorizer on
    training text only, then transform the test text.
7.  **Train the model** using `MultinomialNB`.
8.  **Evaluate the model** using accuracy, a confusion matrix, and
    `classification_report`.
9.  **Predict new reviews** by transforming them with the same fitted
    vectorizer and passing them to the trained model.

## Algorithm Explanation

### Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classification algorithm
commonly used for text classification. With bag-of-words features, it
learns how word counts are associated with each sentiment class and
estimates the most likely class for a review.

### CountVectorizer

`CountVectorizer` builds a vocabulary from the training text and
converts each review into a numerical representation based on word
counts. The trained vectorizer must also be used for future reviews so
that their features match the model's expected input.

## Project Structure

``` text
Customer-Review-Sentiment-Analysis/
├── Customer_Review_Sentiment_Analysis.ipynb
├── customer_review_sentiment_dataset.csv
└── README.md
```

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/mdaasik-tech/Customer-Review-Sentiment-Analysis.git
cd Customer-Review-Sentiment-Analysis
```

> If your repository has a different name, replace the repository URL
> and folder name with your actual GitHub repository details.

### 2. Install dependencies

``` bash
pip install pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Run the notebook

Open `Customer_Review_Sentiment_Analysis.ipynb` in Jupyter Notebook or
upload it to Google Colab.

Make sure the CSV file is available in the notebook's working directory.
If running locally, update the notebook's dataset path from
`/content/customer_review_sentiment_dataset.csv` to the correct local
path, or use a relative path such as:

``` python
df = pd.read_csv("customer_review_sentiment_dataset.csv")
```

Run the cells from top to bottom.

## Evaluation

The notebook reports:

-   **Accuracy** --- overall proportion of correctly classified test
    reviews.
-   **Confusion matrix** --- compares actual sentiment labels with
    predicted labels.
-   **Classification report** --- shows precision, recall, F1-score, and
    support for each class.

The notebook calculates the actual scores when it is run. No fixed
accuracy value is stated here because the notebook file does not include
saved evaluation output.

## Example Predictions

The notebook tests example reviews such as:

-   "Waste of money! Extremely poor durability and slow performance."
-   "Works perfectly right out of the box. Highly recommend to
    everyone!"
-   "Not Good enough"
-   "wast of money"

The model returns a predicted sentiment for each review. Predictions
depend on the training data and vocabulary learned by the vectorizer.

## Key Learnings

-   Preparing and exploring a text dataset with Pandas.
-   Separating model inputs from the target label.
-   Using a stratified train-test split.
-   Converting text into numerical features with `CountVectorizer`.
-   Training a text classifier with `MultinomialNB`.
-   Evaluating a multiclass classification model.
-   Reusing a fitted vectorizer for new text predictions.

## Possible Improvements

-   Add a preprocessing pipeline for consistent text cleaning.
-   Compare `CountVectorizer` with TF-IDF features.
-   Compare Naive Bayes with Logistic Regression or Linear SVM.
-   Review class-wise precision and recall to identify which sentiments
    are harder to predict.
-   Save the trained model and vectorizer for reuse in an application.
-   Build a simple web interface for live review predictions.

## Author

**Mohammed Aasik.A**

-   LinkedIn:
    https://www.linkedin.com/in/mohammed-aasik-aspiring-machine-learning-engineer-257787433/
-   GitHub: https://github.com/mdaasik-tech
