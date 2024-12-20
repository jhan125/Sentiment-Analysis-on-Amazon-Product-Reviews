# Sentiment-Analysis-on-Amazon-Product-Reviews

## Overview

This project aims to develop a sentiment analysis model specifically tailored to Amazon product reviews in the food category. By adapting a sophisticated, pre-trained language model (DistilBERT) to this particular domain, we aim to accurately classify the sentiment polarity (positive, negative, neutral) of customer reviews. The approach leverages advanced Natural Language Processing (NLP) techniques and machine learning methods, while also considering computational efficiency and scalability.

## Objectives

- **Domain-Specific Adaptation:** Fine-tune a pre-trained language model (DistilBERT) on a curated dataset of Amazon food product reviews.
- **Comprehensive Model Exploration:** Evaluate the performance of various models—ranging from Naive Bayes and Logistic Regression to LSTM, CNN, BERT, and RoBERTa—before selecting the most suitable model for the given task.
- **Practical Applicability:** Provide valuable insights into consumer preferences and market trends, empowering sellers and buyers with data-driven decision-making tools.

## Key Features

- **Data Collection & Preprocessing:**  
  Collection of a substantial and diverse dataset of Amazon food category reviews. Preprocessing steps (e.g., tokenization, lemmatization, removal of stopwords) ensure data quality.
  
- **Model Training & Fine-Tuning:**  
  Training multiple models and fine-tuning pre-trained transformers like DistilBERT for sentiment classification.
  
- **Evaluation & Benchmarking:**  
  Measuring model performance using standard metrics (accuracy, F1-score, precision, recall) to identify the most effective approach.
  
- **Resource Utilization:**  
  Efficiently leveraging Google Colab and the NEU Discovery Cluster for computation, ensuring reproducibility and scalability.


## HuggingFace Model

**Model Name:** [distilbert-base-uncased-finetuned-amazon-food-reviews](https://huggingface.co/jhan21/distilbert-base-uncased-finetuned-amazon-food-reviews)

**Model Description:**  
This model is a fine-tuned version of `distilbert-base-uncased` trained on an Amazon food reviews dataset. It classifies the sentiment of the reviews into positive, negative, or neutral.

**Metrics on Evaluation Set:**
- Loss: 0.08
- Accuracy: 0.87
- Precision: 0.71
- Recall: 0.77
- F1: 0.73

**Training Details:**
- **Learning Rate:** 5e-5
- **Train Batch Size:** 8
- **Eval Batch Size:** 8
- **Seed:** 0
- **Optimizer:** Adam (betas=(0.9,0.999), epsilon=1e-08)
- **LR Scheduler:** Linear
- **Epochs:** 5

**Performance by Class:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| -1    | 0.77      | 0.76   | 0.76     | 851      |
| 0     | 0.38      | 0.62   | 0.47     | 467      |
| 1     | 0.97      | 0.92   | 0.94     | 4985     |
| **Accuracy**       |         |        | **0.87**  | 6303     |
| **Macro Avg**      | 0.71    | 0.77   | 0.73     | 6303     |
| **Weighted Avg**   | 0.90    | 0.87   | 0.88     | 6303     |

**Training Process (Selected Checkpoints):**
| Training Loss | Epoch | Step  | Validation Loss | Accuracy | Precision | Recall | F1    |
|---------------|-------|-------|-----------------|----------|-----------|--------|-------|
| 0.3730        | 1.00  | 10000 | 0.3706          | 0.8782   | 0.7040    | 0.7657 | 0.7295 |
| 0.3631        | 2.00  | 20000 | 0.3517          | 0.8805   | 0.7145    | 0.7679 | 0.7226 |
| 0.2913        | 3.00  | 30000 | 0.4759          | 0.8697   | 0.7132    | 0.7653 | 0.7239 |
| 0.2839        | 3.50  | 35000 | 0.4980          | 0.8755   | 0.7166    | 0.7693 | 0.7311 |
| 0.2184        | 4.50  | 45000 | 0.5912          | 0.8888   | 0.7147    | 0.7498 | 0.7310 |
| 0.0891        | 4.85  | 48500 | 0.8237          | 0.8731   | 0.7065    | 0.7651 | 0.7258 |

**Framework Versions:**
- Transformers: 4.35.2
- PyTorch: 2.1.0+cu118
- Tokenizers: 0.15.0

## Note
This project was developed as an independent effort in the CS6120 Natural Language Processing course. It stands as both a practical tool for sentiment analysis and a learning endeavor in the NLP domain.
