import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

st.title("📰 News Topic Classifier (DistilBERT)")

model = DistilBertForSequenceClassification.from_pretrained("./results")
tokenizer = DistilBertTokenizerFast.from_pretrained("./results")

labels = ["World", "Sports", "Business", "Science/Technology"]

text = st.text_area("Enter News Headline")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter a headline.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=1).item()
        st.success(f"Predicted Category: **{labels[prediction]}**")
