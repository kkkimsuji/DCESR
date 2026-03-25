import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def extract_individual_review_emotions(text_list, device, batch_size=1, max_length=512):
    """
    Extracts 7D emotion probability vectors for individual reviews[cite: 211, 296].
    Uses DistilRoBERTa-base fine-tuned for multi-emotion classification[cite: 265, 272].
    """
    model_name = 'j-hartmann/emotion-english-distilroberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    all_emotion_vectors = []

    print(f"[INFO] Extracting emotions from {len(text_list)} reviews...")
    
    for i in tqdm(range(0, len(text_list), batch_size), desc="Emotion Extraction"):
        batch = text_list[i:i + batch_size]
        
        inputs = tokenizer(
            batch, 
            return_tensors='pt', 
            max_length=max_length, 
            truncation=True, 
            padding='max_length'
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Transform 7D logits into a probability distribution via softmax [cite: 288, 296]
            probs = F.softmax(outputs.logits, dim=-1)
            all_emotion_vectors.append(probs.cpu().numpy())

    return np.vstack(all_emotion_vectors)