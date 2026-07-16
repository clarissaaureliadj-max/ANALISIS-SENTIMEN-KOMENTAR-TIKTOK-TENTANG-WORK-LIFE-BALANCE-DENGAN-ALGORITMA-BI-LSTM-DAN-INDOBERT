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

## 🏗️ Project Architecture / Research Workflow

## 🏗️ Project Architecture / Research Workflow

The research follows the **CRISP-DM** methodology. The interactive process workflow framework is detailed below:

```mermaid
graph TD
    %% Defining Color Palette (Clean Executive Vibe)
    classDef stepBox fill:#FCFBF9,stroke:#5C4033,stroke-width:2px,rx:8,ry:8,color:#2C1E18;
    classDef accentBox fill:#F4F1EA,stroke:#8B7355,stroke-width:2px,rx:8,ry:8,color:#2C1E18;
    classDef lineStyle stroke:#8B7355,stroke-width:2px;

    %% Workflow Nodes with Detailed Description Strings
    Step1["<b>1. Business Understanding</b><br/>• Manual evaluation of massive social media text is slow and subjective.<br/>• <i>Goal: Build an automated system to analyze public perception of work-life balance.</i>"]:::stepBox
    
    Step2["<b>2. Data Understanding</b><br/>• Raw TikTok comments scraped primary data source using Apify API.<br/>• Gathered 8,069 raw textual interactions, likes, and workplace opinions."]:::stepBox
    
    Step3["<b>3. Data Preparation</b><br/>• Pipeline: Case folding ➔ emoji & punctuation removal ➔ tokenization ➔ label mapping.<br/>• Dropped 7,111 missing/corrupted rows & 47 text duplicates ➔ 911 clean samples."]:::stepBox
    
    Step4["<b>4. Modeling Phase</b><br/>• Partitioned data via Stratified Sampling with an 80:20 distribution ratio.<br/>• Parallel architectures training: Sequential Bi-LSTM vs Pre-trained IndoBERT."]:::stepBox
    
    Step5["<b>5. Evaluation Metrics</b><br/>• Compared models stability using Precision, Recall, and Macro/Weighted F1-Score.<br/>• IndoBERT established dominance with 96.35% accuracy over Bi-LSTM's 87.50%."]:::stepBox
    
    Step6["<b>6. System Deployment</b><br/>• Embedded the superior IndoBERT engine into an interactive web interface.<br/>• Streamlit UI framework supports manual inputs, bulk CSV upload, and Apify streams."]:::accentBox

    %% Flow Connections
    Step1 ==> Step2
    Step2 ==> Step3
    Step3 ==> Step4
    Step4 ==> Step5
    Step5 ==> Step6

    %% Applying custom connection lines
    linkStyle 0,1,2,3,4 stroke:#8B7355,stroke-width:2px;

### Alternatif Menggunakan File Gambar (`.png`)
Jika Anda tetap ingin mengekspor kode Mermaid di atas menjadi berkas gambar statis menggunakan *tool* seperti [Mermaid Live Editor](https://mermaid.live/) lalu menyimpannya sebagai file `research_workflow.png`, Anda bisa menggunakan tag `<p align="center">` berikut untuk menampilkannya di GitHub secara rapi dan berada di posisi tengah:

```markdown
## 🏗️ Project Architecture / Research Workflow

The research follows the **CRISP-DM** methodology. The framework diagram (in English) is shown below — vector source: `research_workflow.svg`.

<p align="center">
  <img src="data/research_workflow.png" alt="Research Workflow — TikTok WLB Sentiment Analysis" width="85%">
</p>

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

---

## 📊 Dataset Preparation

**Total Initial Samples:** 8,069 raw comment records.  
**Final Filtered Dataset:** 911 samples (after handling systemic missing values and removing 47 duplicate text entries). 
**File Format:** csv.  
**Final Dataset Sentiment Distribution:** Negative: 4,771 comments (59.13%).  Positive: 1.832 comments (22.70%).  Neutral: 1,466 comments (18.17%).  

```
tiktok-wlb-sentiment/
├── data/
│   ├── tiktokcomment.csv            # Raw dataset exported from Apify
│   └── dataset_processed.csv        # Cleaned dataset ready for training
├── models/
│   ├── bilstm_model.h5
│   └── indobert_fine_tuned/

```
---

## 🏋️ Training

The model training process is configured using a Stratified Sampling distribution at an 80:20 ratio (6,455 training samples and 1,614 testing samples).

Structure:
```bash
python train_indobert.py --batch_size 128 --max_length 512 --epochs 5
python train_bilstm.py --epochs 10 --batch_size 32
```

---

## 📊 Results

IndoBERT demonstrated absolute dominance, achieving a peak accuracy of 96.35%. The transformer self-attention mechanism proved significantly more precise at understanding casual, short, and highly variable sentence structures unique to Indonesian TikTok comments.  Bi-LSTM achieved a solid accuracy of 87.50%, showing its best training stability around the 9th epoch before indicating mild overfitting tendencies toward the end of the process

---

## 🏗️ Project Structure

**Section Description** Explain you project structure.

Structure: 

```
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

---

## 📝 Citation

```bibtex
@article{djoniwan2026analisis,
  title={Analisis Sentimen Komentar Tiktok Tentang Work Life Balance Dengan Menggunakan Algoritma Bi-LSTM Dan IndoBERT},
  author={Djoniwan, Clarissa Aurelia},
  journal={Thesis Report of Information Systems Study Program},
  school={Universitas Multimedia Nusantara},
  year={2026}
}
```

---

## 🙏 Acknowledgments

Big Data Lab, Information Systems Study Program, Universitas Multimedia Nusantara (UMN)

---

## 📧 Contact

- Campus Email: clarissa.aurelia1@student.umn.ac

---

## 📜 License

This academic research project is released under the MIT License. The codebase is entirely open-source and welcoming of future developments. See the LICENSE file for more explicit copyright details.

---

## Tips

You can just download this Read Me template and modify it. 

---
