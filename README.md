# Intent-Based Chatbot using Flask and Keras

This project implements an intent-based customer support chatbot using a trained deep learning model and deploys it as a web application using Flask. The chatbot classifies user messages into predefined intents and returns appropriate responses.

---

## Features

- Intent classification using a trained neural network  
- Text preprocessing with tokenization and label encoding  
- Simple web-based chat interface  
- Backend built with Flask  
- Ready for local running and cloud deployment  

---

## Project Structure
```
.
├── app.py
├── intent_model.keras
├── tokenizer.pkl
├── label_encoder.pkl
├── templates/
│ └── index.html
├── requirements.txt
└── README.md
```


- `app.py` – Main Flask application and inference logic  
- `intent_model.keras` – Trained intent classification model  
- `tokenizer.pkl` – Tokenizer used during training  
- `label_encoder.pkl` – Label encoder for intent classes  
- `templates/index.html` – Frontend chat interface  

---

## How It Works

1. User enters a message in the web interface  
2. The message is sent to the Flask backend  
3. Text is preprocessed using the saved tokenizer  
4. The trained model predicts the intent  
5. A predefined response for that intent is returned to the user  

---

## Installation

1. Clone the repository:

```
git clone <your-repo-link>
cd intent-chatbot
```
2. Create and activate a virtual environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Run the Application Locally
```
python app.py
```
5. Then open your browser and go to:
```
http://127.0.0.1:5000/
```

---

## Model Details
  - Input: User text message
  - Preprocessing: Tokenization and padding
  - Model: Keras neural network trained for intent classification
  - Output: Predicted intent label
  
The model, tokenizer, and label encoder used during training are saved and loaded at runtime to ensure consistent predictions.

---

## Intents
Some intents handled by the chatbot:
- Order status
- Refund queries
- Product detail queries
- Technical support
- Fallback
