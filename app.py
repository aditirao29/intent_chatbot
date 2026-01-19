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

emailid = "support@company.com"
phone = "+1-800-123-4567"
order_link = "https://company.com/track-order"
refund_link = "https://company.com/refund"
prod_link = "https://company.com/products"
acc_link = "https://company.com/account"
help_link = "https://company.com/help"
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
        "I can help you check your order status.\n\n"
        "If you have your order ID, you can track it here:\n"
        f"{order_link}\n\n"
        "Orders usually take 2-5 business days to ship. If it's past that, let me know.",

        "Your order may still be processing or already shipped.\n\n"
        "You can view real-time updates using this link:\n"
        f"{order_link}\n\n"
        f"If the status hasn't changed in 48 hours, you can contact {emailid}.",

        "Let me guide you on your delivery.\n\n"
        "• Processing: Order is being prepared\n"
        "• Shipped: On the way\n"
        "• Delivered: Successfully delivered\n\n"
        f"Track here: {order_link}"
    ],

    "refund_query": [
        "Here's a quick summary of our refund policy:\n\n"
        "• Refunds are available within 7 days of delivery\n"
        "• The product must be unused and in original packaging\n"
        "• Refunds are processed within 5-7 business days\n\n"
        f"You can submit a request here:\n{refund_link}",

        "To request a refund, please follow these steps:\n\n"
        "1. Go to your orders section\n"
        "2. Select the item\n"
        "3. Click on 'Request Refund'\n\n"
        "Once approved, the amount is credited back to your original payment method.",

        "I understand refunds can be urgent.\n\n"
        "Refunds are accepted within 7 days after delivery if the item is unused.\n"
        "After approval, the amount is credited within 5-7 working days.\n\n"
        f"If you face issues, contact us at {emailid}."
    ],

    "product_details": [
        "Here's what you can usually find about our products:\n\n"
        "• Price and offers\n"
        "• Technical specifications\n"
        "• Warranty details\n\n"
        f"Browse products here:\n{prod_link}",

        "If you're looking for size, color, features, or compatibility details,\n"
        f"you can check the full product page here:\n{prod_link}\n\n"
        "If something is unclear, I can help explain it.",

        "Each product page contains:\n\n"
        "• Description\n"
        "• Specifications\n"
        "• Customer reviews\n\n"
        f"Visit: {prod_link}"
    ],

    "tech_support": [
        "For login or OTP issues, try these steps first:\n\n"
        "• Check your internet connection\n"
        "• Wait 30 seconds before requesting a new OTP\n"
        "• Make sure your phone number is correct\n\n"
        f"If it still fails, visit: {acc_link}",

        "If the app or website is not responding:\n\n"
        "• Refresh the page\n"
        "• Clear browser cache\n"
        "• Try again after a few minutes\n\n"
        "Temporary server issues usually resolve quickly.",

        "Technical issues can happen sometimes.\n\n"
        "If basic troubleshooting doesn't work, our technical team can help you.\n"
        f"Email: {emailid}\n"
        f"Phone: {phone}"
    ],

    "fallback": [
        "I can help you with:\n\n"
        "• Order tracking\n"
        "• Refunds\n"
        "• Product details\n"
        "• Technical issues\n\n"
        "Please tell me what you're facing.",

        "I'm here to assist you with your order, payment, or technical problems.\n\n"
        "Could you describe your issue in a bit more detail?",

        "I didn't fully understand that.\n\n"
        "You can ask about order status, refunds, products, or login issues."
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
    options = responses.get(intent, responses["fallback"])
    return random.choice(options)

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
    app.run(host="0.0.0.0",port=5000)
