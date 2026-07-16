# Project Title

Sentiment Analysis of TikTok Comments on Work-Life Balance Using Bi-LSTM and IndoBERT Algorithms
---

## 📋 Overview

This project is an experimental implementation of a thesis research focused on analyzing digital public opinion regarding the issue of Work-Life Balance. Given the critical impact of work balance on employee productivity and mental health in Indonesia, the TikTok social media platform was explored as the primary medium where the public openly expresses their firsthand views.  The main objective of this project is to identify the distribution and classification of public sentiment and to conduct a deep performance comparison between two popular deep learning architectures in Indonesian text processing: Bidirectional Long Short-Term Memory (Bi-LSTM) and Indonesian Bidirectional Encoder Representations from Transformers (IndoBERT). The data mining development framework applied throughout this project is based on the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology.  

### Key Features

**Automatic Scraper Integration:** Automated data extraction from the TikTok platform utilizing automated API integration via Apify.  **Advanced Text Preprocessing:** A comprehensive pipeline for cleaning informal Indonesian text, which includes case folding, text cleaning (removal of emojis, symbols, and punctuation), and subword tokenization.  
**State-of-the-art Model Evaluation:** An in-depth performance comparison between a two-way sequential approach (Bi-LSTM) and a local Transformer-based architecture (Pre-trained IndoBERT). 
**Aspect-Based Insight Identification:** Categorization of discussions focusing on five core aspects of work-life balance: workload, flexibility, mental health, salary/compensation, and work environment.  
**Interactive Multi-Mode Deployment:** A user-friendly web interface dashboard developed with Streamlit that supports text analysis via manual text entry, bulk CSV file uploads, or real-time dataset pulling using an Apify dataset ID.  

---

## 🎯 Key Contributions

**Work-Life Balance Analysis Benchmark:** Provides empirical distribution data regarding Indonesian public sentiment on social media towards current employment and workplace dynamics.  
**Indonesian NLP Model Comparison:** Generates a high-quality comparative performance analysis between Deep Learning vs. Transformer Engines specifically trained on informal/casual conversational text corpora.  
**Optimized Classification Pipeline:** Establishes a structured pipeline workflow ranging from handling system-level missing values caused by scraping to stratified dataset split data modeling.  
**Integrated Practical Dashboard:** Provides a ready-to-use application prototype for HR practitioners and corporate policymakers to monitor public work satisfaction trends in real-time. 

---

## Project Architecture / Research Workflow

[Business Understanding] ──> [Data Understanding (Scraping/Apify)]
                                       │
                                       ▼
[Modeling (Bi-LSTM / IndoBERT)] <── [Data Preparation (Cleaning & Tokenization)]
               │
               ▼
   [Evaluation Metrics] ──> [Deployment (Streamlit App)]

---

## 🚀 Installation

### Requirements

Python 3.8+ , PyTorch 2.0+ , Transformers (Hugging Face) , Scikit-Learn , Streamlit , Pandas & NumPy , Matplotlib & Seaborn , TextBlob & WordCloud

### Setup

1. Clone the repository:
```bash
git clone https://github.com/clarissaaurelia/tiktok-wlb-sentiment.git
cd tiktok-wlb-sentiment
```

2. Create a conda environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt

```

## 📊 Dataset Preparation

**Total Initial Samples:** 8,069 raw comment records.  
**Final Filtered Dataset:** 911 samples (after handling systemic missing values and removing 47 duplicate text entries).  
**File Format:** csv.  
**Final Dataset Sentiment Distribution:** Negative: 4,771 comments (59.13%). Positive: 1.832 comments (22.70%). Neutral: 1,466 comments (18.17%).  

tiktok-wlb-sentiment/
├── data/
│   ├── tiktokcomment.csv            # Raw dataset exported from Apify
│   └── dataset_processed.csv        # Cleaned dataset ready for training
├── models/
│   ├── bilstm_model.h5
│   └── indobert_fine_tuned/

```

## 🏋️ Training

The model training process is configured using a Stratified Sampling distribution at an 80:20 ratio (6,455 training samples and 1,614 testing samples). To train the IndoBERT model (fine-tuned on top of the base weights of w11wo/indonesian-roberta-base-sentiment-classifier):

python train_indobert.py --batch_size 128 --max_length 512 --epochs 5```
python train_bilstm.py --epochs 10 --batch_size 32
```

---

## 📊 Results

MetricIndoBERT ModelBi-LSTM ModelAccuracy96.35%87.50%Precision Macro94.89%75.10%Recall Macro93.11%67.52%F1-Score Macro93.93%70.74%Precision Weighted96.42%86.66%Recall Weighted96.35%87.50%F1-Score Weighted96.37%86.87%

---

## 🏗️ Project Structure

tiktok-wlb-sentiment/
├── data/                             # Internal dataset files
├── src/
│   ├── preprocessing.py              # Text cleaning and case folding pipelines
│   ├── loader.py                     # DataLoader configurations & Subword Tokenizer
│   └── models/
│       ├── indobert_classifier.py    # IndoBERT HuggingFace model architecture module
│       └── bilstm_classifier.py      # Bi-LSTM Layer architecture module
├── app.py                            # Streamlit web dashboard application code
├── train_indobert.py                 # Training script for IndoBERT model
├── train_bilstm.py                   # Training script for Bi-LSTM model
├── requirements.txt                  # Project software dependency list
└── README.md
```

```

## 📝 Citation

  ```bibtex
  title={Analisis Sentimen Komentar Tiktok Tentang Work Life Balance Dengan Menggunakan Algoritma Bi-LSTM Dan IndoBERT},
  author={Djoniwan, Clarissa Aurelia},
  journal={Thesis Report of Information Systems Study Program},
  school={Universitas Multimedia Nusantara},
  year={2026}
}
```

```

## 🙏 Acknowledgments

Big Data Lab, Information Systems Study Program, Universitas Multimedia Nusantara (UMN)

---

## 📧 Contact

Campus Email: clarissa.aurelia1@student.umn.ac.id

---

## 📜 License

This academic research project is released under the MIT License. The codebase is entirely open-source and welcoming of future developments. See the LICENSE file for more explicit copyright details.

---

## Tips

You can just download this Read Me template and modify it. 
---
