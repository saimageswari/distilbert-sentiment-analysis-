<img width="1280" height="716" alt="WhatsApp Image 2026-08-31 at 10 38 41" src="https://github.com/user-attachments/assets/e50f19a4-edcb-4eea-975a-e4ad189a9abd" />
# 🧠 DistilBERT Sentiment Analysis

An end-to-end NLP application that uses a **fine-tuned DistilBERT Transformer model** to classify movie reviews as **Positive** or **Negative**.

The trained model is hosted on **Hugging Face**, while the application code is maintained on **GitHub** and served through an interactive **Streamlit** interface.

---

## 🚀 Project Overview

This project demonstrates how a pretrained Transformer model can be fine-tuned for a real-world Natural Language Processing classification task.

The application accepts a movie review as input and uses the fine-tuned DistilBERT model to predict:

* 😊 Positive sentiment
* 😞 Negative sentiment
* 🎯 Prediction confidence
* 📊 Sentiment probabilities

The Streamlit interface includes an interactive visualization of the Transformer inference process.

---
<img width="1280" height="707" alt="WhatsApp Image 2026-08-31 at 10 38 21" src="https://github.com/user-attachments/assets/ee2d8948-9ed5-45f3-9f97-7a9556d8ca



## ✨ Features

* 🤗 Fine-tuned DistilBERT Transformer
* 🎬 IMDb movie review sentiment classification
* 🧠 Binary text classification
* 📊 Accuracy and F1-score evaluation
* 🎯 Confidence score visualization
* 📈 Sentiment probability chart
* 🌐 Interactive Streamlit web application
* 🧊 Interactive 3D AI visualization
* ☁️ Model hosted on Hugging Face
* 💻 Application source code hosted on GitHub

---

## 🏗️ Architecture

```text
                 IMDb Dataset
                      │
                      ↓
              Text Preprocessing
                      │
                      ↓
             DistilBERT Tokenizer
                      │
                      ↓
             Pretrained DistilBERT
                      │
                      ↓
                 Fine-Tuning
                      │
                      ↓
              Sentiment Classifier
                      │
             ┌────────┴────────┐
             ↓                 ↓
        Hugging Face       Streamlit
          Model              App
             │                 │
             └────────┬────────┘
                      ↓
              Sentiment Result
                      │
             ┌────────┴────────┐
             ↓                 ↓
         Prediction         Confidence
          Label             Probability
```

---

## 🤖 Model

**Model:** DistilBERT

DistilBERT is a lightweight Transformer model derived from BERT. It provides a good balance between language understanding and computational efficiency.

### Fine-Tuning Configuration

| Parameter               | Value                     |
| ----------------------- | ------------------------- |
| Base Model              | DistilBERT                |
| Task                    | Sequence Classification   |
| Dataset                 | IMDb                      |
| Number of Labels        | 2                         |
| Maximum Sequence Length | 256                       |
| Training Epochs         | 2                         |
| Framework               | PyTorch                   |
| Library                 | Hugging Face Transformers |

### Labels

```text
0 → NEGATIVE
1 → POSITIVE
```

---

## 📊 Model Performance

The fine-tuned model achieved the following results on the evaluation dataset:

| Metric          |  Score |
| --------------- | -----: |
| Accuracy        | 91.41% |
| F1 Score        | 91.44% |
| Evaluation Loss | 0.2609 |

These results demonstrate that the fine-tuned Transformer can effectively distinguish between positive and negative movie reviews.

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning / NLP

* PyTorch
* Hugging Face Transformers
* DistilBERT

### Dataset

* IMDb Movie Reviews

### Web Application

* Streamlit
* Plotly
* Three.js

### Development Tools

* Google Colab
* VS Code
* Git
* GitHub
* Hugging Face

---

## 📂 Project Structure

```text
distilbert-sentiment-analysis/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The trained model is hosted separately on Hugging Face rather than being stored in the GitHub repository.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/saimageswari/distilbert-sentiment-analysis-.git
```

Navigate to the project:

```bash
cd distilbert-sentiment-analysis-
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start Streamlit:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 🔍 Example

### Input

```text
This movie was absolutely fantastic. 
The acting was brilliant and the story was amazing.
```

### Output

```text
😊 POSITIVE

Confidence: 98%
```

---

## 🧠 How Prediction Works

The application follows this process:

```text
User enters review
        ↓
DistilBERT Tokenizer
        ↓
Input IDs + Attention Mask
        ↓
Fine-tuned DistilBERT
        ↓
Logits
        ↓
Softmax
        ↓
Class Probability
        ↓
Positive / Negative
```

The predicted class is selected using the highest probability.

---

## 🌐 Model Hosting

The fine-tuned DistilBERT model is hosted on Hugging Face.

**Model Repository:**

`saimagesh/distilbert-sentiment-final`

Replace the above with your actual Hugging Face model repository.

---

## 📈 Future Improvements

* Add neutral sentiment classification
* Add emotion classification
* Add batch prediction for CSV files
* Add confusion matrix visualization
* Add model comparison with BERT
* Add attention visualization
* Deploy the application publicly
* Add multilingual sentiment analysis

---

## 👨‍💻 Author

**Sai Mageswari**

BCA Student | Machine Learning & Generative AI Enthusiast

---

## ⭐ If you found this project useful

Feel free to explore the repository, experiment with the model, and improve the application.
