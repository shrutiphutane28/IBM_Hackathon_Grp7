# Vulnerability Detection Using RoBERTa

This project leverages the RoBERTa model to detect and classify vulnerabilities in JavaScript code. The dataset is processed using Huggingface Transformers and PyTorch, and vulnerabilities are categorized based on `vulnType`.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Details](#dataset-details)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Results](#results)
8. [Future Enhancements](#future-enhancements)

## Introduction

This project automates the identification of vulnerabilities in JavaScript code snippets using machine learning. The system processes text descriptions of vulnerabilities, tokenizes the data, and fine-tunes the RoBERTa model for classification.

## Dataset Details

The dataset is hosted on IBM Cloud Object Storage and contains manually confirmed vulnerabilities.

**Key columns in the dataset:**
- `details`: Description of the vulnerability.
- `vulnType`: The type of vulnerability (target label).
- `versions` and `CVE`: Metadata for additional context.

## Features

- **Preprocessing**: Handles missing values, cleans the dataset, and prepares inputs for training.
- **Fine-Tuning**: Custom training of RoBERTa to classify vulnerabilities.
- **Web Integration**: Can be integrated into a web application for real-time detection.
- **Evaluation**: Includes metrics like precision, recall, and F1-score for validation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vulnerability-detection.git
   cd vulnerability-detection
   ```
2. Install dependencies:
   ```bash
   pip install --upgrade accelerate pandas transformers torch scikit-learn
   ```
3. Ensure access to the IBM Cloud Object Storage for dataset retrieval.

## Usage

### 1. Preprocessing

- Loads the dataset from IBM Cloud Object Storage.
- Cleans and tokenizes the data.

### 2. Model Training

Train the model using the following command:
```python
trainer.train()
```

### 3. Inference

To predict vulnerabilities in new JavaScript code snippets:
```python
new_data = ["const userInput = \"<script>alert('XSS Attack');</script>\"; ..."]
inputs = tokenizer(new_data, padding=True, truncation=True, return_tensors="pt", max_length=128)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1)
print(f"Predicted vulnerability type: {list(label_map.keys())[predicted_class]}")
```

## Model Training and Evaluation

- The model is trained on the processed dataset for 3 epochs using the Huggingface Trainer API.
- Evaluation is performed on a validation set using metrics like precision, recall, and F1-score.
- A classification report is generated to summarize performance.

## Results

- The trained model accurately classifies vulnerabilities with high precision and recall.
- Sample classification results are displayed using `classification_report` from Scikit-Learn.

## Future Enhancements

- Incorporate CodeBERT for improved results.
- Extend support to multiple programming languages.
- Develop a user-friendly web interface for real-time vulnerability detection.

