import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping

def save_split_data(df, path):
    """
    Saves the data splits into Parquet format for reproducibility.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, compression='snappy')
    print(f"[INFO] Data split saved to: {path}")

def prepare_dcesr_inputs(df):
    """
    Prepares the 4-channel input (User Semantic, Item Semantic, User Emotion, Item Emotion)
    and the target ratings as numpy arrays.
    """
    # Stack list-type columns into (Batch, Dimension) numpy arrays
    user_sem = np.stack(df['user_semantic'].values)
    item_sem = np.stack(df['item_semantic'].values)
    user_emo = np.stack(df['user_emotion'].values)
    item_emo = np.stack(df['item_emotion'].values)
    
    # Ensure ratings are in float32 for TensorFlow compatibility
    y = df['rating'].values.astype('float32')
    
    return [user_sem, item_sem, user_emo, item_emo], y

def run_training_pipeline(model, config):
    """
    Executes the training and evaluation workflow using the provided model instance.
    """
    # --- 1. Load Preprocessed Data ---
    data_path = config['data']['output_path']
    print(f"\n[TRAIN] Loading processed data from: {data_path}")
    df = pd.read_pickle(data_path)

    # --- 2. Data Splitting ---
    # Split into 7:1:2 (Train:Val:Test) based on config ratios
    test_size = config['model'].get('test_size', 0.2)
    val_size = config['model'].get('val_size', 0.125)
    
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=42)
    
    print(f"[INFO] Split Statistics - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # --- 3. Save Splits for Consistency ---
    save_split_data(train_df, config['data']['train_path'])
    save_split_data(val_df, config['data']['val_path'])
    save_split_data(test_df, config['data']['test_path'])

    # --- 4. Prepare Model Inputs ---
    X_train, y_train = prepare_dcesr_inputs(train_df)
    X_val, y_val = prepare_dcesr_inputs(val_df)
    X_test, y_test = prepare_dcesr_inputs(test_df)

    # --- 5. Model Compilation ---
    print("[TRAIN] Compiling the model with Adam optimizer...")
    lr = config['model'].get('learning_rate', 0.0002)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )

    # --- 6. Training with EarlyStopping ---
    patience = config['model'].get('patience', 5)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=patience,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    batch_size = config['model'].get('batch_size', 128)
    epochs = config['model'].get('epochs', 50)

    print("\n--- Starting Model Training ---")
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    # --- 7. Final Performance Evaluation ---
    print("\n--- Calculating Final Test Metrics ---")
    y_pred = model.predict(X_test).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"Test Result -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")

    # --- 8. Save Final Model Weights ---
    save_path = config.get('model_save_path', 'model/dcesr_final_model.h5')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"[SUCCESS] Best model weights saved to: {save_path}")
    
    return history