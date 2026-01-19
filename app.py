from flask import Flask,render_template,request,jsonify
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

app = Flask(__name__)

model = tf.keras.models.load_model("intent_model.keras")
with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl","rb") as f:
    le = pickle.load(f)

def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]","",text)
    text = re.sub(r"\s+"," ",text).strip()
    return text

email = "support@yourcompany.com"
phone = "+1-800-123-4567"
order_link = "https://yourcompany.com/track-order"
refund_link = "https://yourcompany.com/refund"
prod_link = "https://yourcompany.com/products"
acc_link = "https://yourcompany.com/account"
help_link = "https://yourcompany.com/help"
refund_keywords = {"refund","money","return","returned","cash","credit","credited","amount","repay","reimbursement","reverse","reversed","chargeback"}
order_keywords = {"order","track","delivery","arrive","arrived","shipped","dispatch","dispatched","package","status"}
keywords = {"order","track","delivery","arrive","arrived","package","dispatch","dispatched","shipped","refund","money","return","credited",
    "otp","account","login","password","price","cost","warranty","product","website","loading","error","support","help","model","details", 
    "information","item","app","features","specifications","logging","material","issue","problem","wrong","failed","not","working","cant","cannot",
    "unable","crash","crashed","stuck","hang","hanging","slow","bug","glitch"}

order = le.transform(["order_status"])[0] if "order_status" in le.classes_ else None
refund = le.transform(["refund_query"])[0] if "refund_query" in le.classes_ else None
responses = {
    "order_status": [
        "You can track the current status of your order using this link:\n"
        f"{order_link}\n\n"
        "If your order is delayed or the status has not updated, you may:\n"
        f"• Contact our support team at {email}\n"
        f"• Call us at {phone} for immediate assistance."
    ],

    "refund_query": [
        "You can review our refund policy and submit a refund request here:\n"
        f"{refund_link}\n\n"
        "If you are eligible, you will receive an email confirmation within 24-48 hours.\n"
        f"For further assistance, contact {email} or call {phone}."
    ],

    "product_details": [
        "You can find detailed information, specifications, and pricing for our products here:\n"
        f"{prod_link}\n\n"
        f"If you need further help, contact {email} or visit {help_link}."
    ],

    "tech_support": [
        "For technical issues related to login, OTP, or app problems, please visit:\n"
        f"{acc_link}\n\n"
        "If the problem continues, you may:\n"
        f"• Email our technical team at {email}\n"
        f"• Call {phone} for urgent support."
    ],

    "fallback": [
        "I can assist you with order tracking, refunds, product information, or technical support.\n\n"
        f"Please visit our help center at {help_link}\n"
        f"Or contact us at {email}."
    ]
}

def predict_intent(text,min_conf=0.35,margin=0.15):
    text_clean = clean(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    pad = pad_sequences(seq,maxlen=25,padding="post",truncating="post")
    probs = model.predict(pad,verbose=0)[0]
    top1_idx = np.argmax(probs)
    top1_prob = float(probs[top1_idx])
    intent = le.inverse_transform([top1_idx])[0]
    words = set(text_clean.split())
    if order is not None and refund is not None:
        order_prob = probs[order]
        refund_prob = probs[refund]
        has_refund_word = len(words&refund_keywords)>0
        has_order_word = len(words&order_keywords)>0
        if abs(order_prob-refund_prob)<margin:
            if has_refund_word:
                intent = "refund_query"
                top1_prob = float(refund_prob)
            elif has_order_word:
                intent = "order_status"
                top1_prob = float(order_prob)
    has_domain_word = len(words & keywords)>0
    if not has_domain_word or top1_prob<min_conf:
        return {"intent":"fallback","confidence":top1_prob}
    return {"intent":intent,"confidence":top1_prob}

def generate_response(intent):
    if intent not in responses:
        return "I'm not sure how to help with that."
    return random.choice(responses[intent])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    data = request.json
    user_text = data.get("message","")
    result = predict_intent(user_text)
    reply = generate_response(result["intent"])
    return jsonify({"intent":result["intent"],"confidence":round(result["confidence"],3),"response":reply})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
