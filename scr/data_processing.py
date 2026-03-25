import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import torch

# Import the feature extraction modules
from scr.bart import extract_bart_embeddings
from scr.distilroberta import extract_individual_review_emotions

def load_and_preprocess(file_path):
    """
    Load raw Amazon review data and perform basic cleaning and 5-core filtering.
    """
    # Load Dataset (Using lines=True for Amazon JSONL format)
    df = pd.read_json(file_path, compression='gzip', lines=True)
    
    # Select essential columns and rename for consistency
    df = df[['reviewerID', 'asin', 'reviewText', 'overall']]
    df.columns = ['user', 'item', 'review', 'rating']
    
    # Basic Data Cleaning
    df = df.drop_duplicates()
    df['review'] = df['review'].fillna('') 
    
    # 5-core Filtering: keeps users/items with at least 5 reviews
    user_counts = df['user'].value_counts()
    item_counts = df['item'].value_counts()
    df = df[df['user'].isin(user_counts[user_counts >= 5].index)]
    df = df[df['item'].isin(item_counts[item_counts >= 5].index)]
    
    # Label Encoding: Convert string IDs into numerical indices
    le_user, le_item = LabelEncoder(), LabelEncoder()
    df['user'] = le_user.fit_transform(df['user'])
    df['item'] = le_item.fit_transform(df['item'])
    
    return df

def generate_review_sets(df):
    """
    Aggregate individual reviews into User Review Sets and Item Review Sets.
    These sets are used as input for the BART semantic channel.
    """
    # Aggregate all reviews written by each user (User Preference)
    user_reviews = df.groupby('user')['review'].apply(lambda x: " ".join(x)).reset_index()
    user_reviews.columns = ['user', 'user_review_set']
    
    # Aggregate all reviews written for each item (Item Attributes)
    item_reviews = df.groupby('item')['review'].apply(lambda x: " ".join(x)).reset_index()
    item_reviews.columns = ['item', 'item_review_set']
    
    # Merge aggregated review sets back into the original dataframe
    df = pd.merge(df, user_reviews, on='user', how='left')
    df = pd.merge(df, item_reviews, on='item', how='left')
    
    return df

def run_preprocessing(config):
    """
    Main entry point for the preprocessing pipeline called by main.py.
    """
    input_path = config['data']['input_path']
    output_path = config['data']['output_path']
    
    # Hardware device (passed from main.py)
    device = torch.device(config['device'])

    # [Step 1] Loading and Initial Processing
    print("\n[PREPROCESS] Step 1: Loading and Filtering Data...")
    df = load_and_preprocess(input_path)
    df = generate_review_sets(df)

    # [Step 2] BART Semantic Channel (768D)
    print(f"\n[PREPROCESS] Step 2: Extracting BART Semantic Features (Max Length: {config['model']['bart_max_length']})")
    
    # Extract User Semantic Features
    unique_u = df.groupby('user')['user_review_set'].first()
    u_sem = extract_bart_embeddings(
        unique_u.tolist(), 
        device, 
        batch_size=config['model']['batch_size'], 
        max_length=config['model']['bart_max_length']
    )
    df['user_semantic'] = df['user'].map(dict(zip(unique_u.index, u_sem)))

    # Extract Item Semantic Features
    unique_i = df.groupby('item')['item_review_set'].first()
    i_sem = extract_bart_embeddings(
        unique_i.tolist(), 
        device, 
        batch_size=config['model']['batch_size'], 
        max_length=config['model']['bart_max_length']
    )
    df['item_semantic'] = df['item'].map(dict(zip(unique_i.index, i_sem)))

    # [Step 3] Emotion Feature Extraction & Averaging (7D)
    print(f"\n[PREPROCESS] Step 3: Extracting and Averaging Emotion Features (Max Length: {config['model']['emotion_max_length']})")
    
    # 3-1: Extract raw 7D vectors for every individual review
    raw_emo = extract_individual_review_emotions(
        df['review'].tolist(), 
        device, 
        batch_size=config['model']['batch_size'], 
        max_length=config['model']['emotion_max_length']
    )
    df['temp_emo'] = list(raw_emo)

    # 3-2: Calculate User Emotion Average (e_u)
    print("[INFO] Computing User Emotion profiles (e_u)...")
    u_map = df.groupby('user')['temp_emo'].apply(lambda x: np.mean(np.stack(x), axis=0)).to_dict()
    df['user_emotion'] = df['user'].map(u_map)

    # 3-3: Calculate Item Emotion Average (e_v)
    print("[INFO] Computing Item Emotion profiles (e_v)...")
    i_map = df.groupby('item')['temp_emo'].apply(lambda x: np.mean(np.stack(x), axis=0)).to_dict()
    df['item_emotion'] = df['item'].map(i_map)

    # Remove temporary raw emotion column
    df.drop(columns=['temp_emo'], inplace=True)

    # [Step 4] Save Final Processed Dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_pickle(output_path)
    
    print(f"\n[SUCCESS] Preprocessing finished! Data points: {len(df)}")
    print(f"[SUCCESS] Processed file saved at: {output_path}")