# Sentiment Analysis with Unigram Logistic Regression


## Project Overview

This project focuses on building a sentiment analysis classifier using unigram logistic regression. The classifier is designed to predict the sentiment (positive or negative) of movie reviews based on preprocessed textual data.

## Key Features

1. **Advanced Preprocessing Techniques**:
   - Tokenization and lowercasing of words.
   - Negation tagging to handle nuanced sentiment expressions (e.g., distinguishing "not good" from "very good").

2. **Feature Dictionary Construction and Normalization**:
   - Creation of a feature dictionary for unigram features, mapping unique words to positions in the feature vector.
   - Min-max normalization of feature values to ensure uniformity and optimize model performance.

3. **Logistic Regression Classifier**:
   - Utilization of Scikit-Learn's `LogisticRegression` model for training.
   - Evaluation of model performance using precision, recall, and F-measure metrics.

## Project Components

### Functions

- **load_corpus(corpus_path)**: Loads a training or test corpus from a specified path and returns a list of (string, int) tuples.

- **is_negation(word)**: Checks if a word is a negation word and returns a boolean.

- **tag_negation(snippet)**: Adds negation tagging to a snippet and returns a list of strings.

- **get_feature_dictionary(corpus)**: Constructs a feature dictionary from the corpus and returns a dictionary {word: index}.

- **vectorize_snippet(snippet, feature_dict)**: Converts a snippet into a feature vector and returns a Numpy array.

- **vectorize_corpus(corpus, feature_dict)**: Converts the corpus into feature vectors and returns a tuple (X, Y) where X and Y are Numpy arrays.

- **normalize(X)**: Performs min-max normalization on feature values.

- **train(corpus_path)**: Trains a logistic regression model on a training corpus and returns the trained model and feature dictionary.

- **evaluate_predictions(Y_pred, Y_test)**: Calculates precision, recall, and F-measure for predictions and returns a tuple of floats.

- **test(model, feature_dict, corpus_path)**: Evaluates a model on a test corpus and prints the results.

- **get_top_features(logreg_model, feature_dict, k=1)**: Selects the top k highest-weight features of a logistic regression model.

## How to Run

### Requirements

- Python 3.x
- Scikit-Learn
- Numpy
- NLTK
