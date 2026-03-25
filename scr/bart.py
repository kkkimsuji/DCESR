import pandas as pd
import numpy as np
import torch
from transformers import BartTokenizer, BartModel
from tqdm import tqdm

def extract_bart_embeddings(text_list, device, batch_size=1, max_length=512):
    """
    Extracts semantic features using the BART encoder's last hidden state.
    Follows DCESR paper logic: Selects the last token's hidden state as the 
    condensed representative summary[cite: 335, 341].
    """
    model_name = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    
    # Process the text list directly provided by the orchestrator
    for i in tqdm(range(0, len(text_list), batch_size), desc="BART Embedding"):
        batch_reviews = text_list[i:i + batch_size]
        
        inputs = tokenizer(
            batch_reviews, 
            return_tensors='pt', 
            max_length=max_length, 
            truncation=True, 
            padding='max_length'
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            
            # Select the last token: (batch, seq, 768) -> (batch, 768)
            # Taking [0] assumes batch_size=1 for precise extraction [cite: 341, 342]
            result = outputs.last_hidden_state[:, -1, :][0].cpu().numpy()
            embeddings.append(result)

    return np.vstack(embeddings)