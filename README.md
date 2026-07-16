# Project Title

Sentiment Analysis of TikTok Comments on Work-Life Balance Using the Bi-LSTM and IndoBERT Algorithms

---

## 📋 Overview

This thesis/project focuses on the automated sentiment analysis of social media comments on TikTok regarding the topic of work-life balance. Given the high volume of unstructured opinions shared by the digital community about workplace pressure, emotional well-being, and time management, manual analysis has become ineffective. This project compares two leading deep learning architectures: Bidirectional Long Short-Term Memory (Bi-LSTM) and IndoBERT (the Indonesian version of Bidirectional Encoder Representations from Transformers). Utilizing the CRISP-DM methodology, this research aims to extract public perceptions and compare the accuracy performance of multi-class classification (Positive, Neutral, and Negative).  

### Key Features

Multi-Class Local Content Sentiment Analysis: Classifies informal expressions of Indonesian users on TikTok into Positive, Neutral, and Negative classes.  
Comprehensive Bidirectional Processing: Utilizes Bi-LSTM to capture sequential word dependencies from both forward and backward directions.  
IndoBERT-Based Self-Attention Mechanism: Leverages the capabilities of the pre-trained Indonesian Transformer model to understand complex semantic contexts and social media slang simultaneously.  
End-to-End CRISP-DM Implementation: A structured pipeline flow starting from raw API data extraction, data preprocessing, architecture modeling, to evaluation metric visualization.  

---

## 🎯 Key Contributions

Deep Learning Model Comparative Study: Provides an empirical comparative study between sequential RNN-based architecture (Bi-LSTM) and Transformer-based architecture (IndoBERT) on informal Indonesian text domains.  
Real-Time Work Policy Aspect Extraction: Identifies crucial keyword visualizations related to workload, clock-out flexibility, mental health, stress levels, as well as salary and compensation from the perspective of digital workers.  
State-of-the-Art (SOTA) Accuracy Achievement: The fine-tuned IndoBERT model successfully achieved a significant performance advantage, reaching a classification accuracy of 96.35%.  

---

## Project Architecture / Research Workflow

[Business Understanding] -> Analyze work-life balance issues & define objective targets
           │
           ▼
[Data Understanding]     -> Collect raw data (8,069 comment samples) via Web API / Apify
           │
           ▼
[Data Preparation]       -> Case Folding, Text cleaning, Tokenization, Sentiment labeling, Split (80:20)
           │
           ▼
[Modeling]               -> Train model architectures in parallel via Bi-LSTM & IndoBERT
           │
           ▼
[Evaluation]             -> Measure performance metrics (Accuracy, Precision, Recall, F1-Score)
           │
           ▼
[Deployment]             -> Integrate the best model into an interactive Streamlit web dashboard

<p align="center">
  <img src="examplearchitecture.png" alt="Architecture of Your Project">
</p>

---

## 🚀 Installation

Python 3.8+  
PyTorch 2.0+Transformers (Hugging Face)  
Scikit-Learn  
Pandas & NumPy  
Streamlit  

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tiktok-wlb-sentiment.git
cd tiktok-wlb-sentiment
```

2. Create a conda environment:
```bash
conda create -n wlb-sentiment python=3.9
conda activate wlb-sentiment
```

3. Install dependencies:
```bash
pip install -r requirements.txt```

---

## 📊 Dataset Preparation

Total Raw Data: 8,069 data rows
Total After Data Cleaning: 911 unique data rows that are clean from empty values (missing values) and duplicate data.
Class Structure: Positive (2), Neutral (1), Negative (0).  

### Download

YourProject/
├── positive_review2000/
│   ├── train/
│   └── test/
├── negative_review2000/
│   ├── train/
│   └── test/
├── neutral_review1000/
│   ├── train/
│   └── test/

```
---

## 🏋️ Training

python train_indobert.py --epochs 3 --batch_size 16 --lr 2e-5
python train_bilstm.py --epochs 10 --batch_size 32 --embedding_dim 128
```

---

## 📊 Results

The Transformer-based model (IndoBERT) consistently dominates all evaluation metrics. This absolute advantage is due to the capability of the Transformer architecture to recognize complex semantic relationships, sarcasm, abbreviations, and the linguistic context of informal Indonesian text much more superiorly than sequential RNN models like Bi-LSTM.

## 🏗️ Project Structure

tiktok-wlb-sentiment/
├── data/
│   ├── tiktokcomment.csv     # Initial raw data
│   └── dataset_final.csv     # Processed ready-to-use data
├── notebooks/
│   ├── eda_and_cleaning.ipynb # Data Exploration & WordCloud
│   ├── indobert_training.ipynb# IndoBERT modeling implementation
│   └── bilstm_training.ipynb  # Bi-LSTM modeling implementation
├── models/
│   ├── best_indobert_model.pt # Best IndoBERT model weights
│   └── best_bilstm_model.h5   # Best Bi-LSTM model weights
├── app.py                     # Streamlit web dashboard interface app
├── utils.py                   # Helper functions for text cleaning & tokenization
└── requirements.txt           # Library dependencies
```

---

## 📝 Citation

@thesis{djoniwan2026analisis,
  title={Analisis Sentimen Komentar Tiktok Tentang Work Life Balance Dengan Menggunakan Algoritma Bi-LSTM Dan IndoBERT},
  author={Djoniwan, Clarissa Aurelia},
  school={Universitas Multimedia Nusantara},
  year={2026}
}
```

---

## 🙏 Acknowledgments

Big Data Lab, Information Systems Study Program, Universitas Multimedia Nusantara (UMN)

---

## 📧 Contact

Maintainer: Clarissa Aurelia Djoniwan
Email: clarissa.aurelia@student.umn.ac.id
---

## 📜 License

This academic research project is licensed under the terms of the MIT License. See the LICENSE document for more complete information.

---
