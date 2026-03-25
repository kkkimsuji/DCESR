import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Dropout
from tensorflow.keras.models import Model

def ProposedModel(config):
    """
    Proposed DCESR-based Recommendation Model.
    Fuses Semantic features (BART) and Emotional features (DistilRoBERTa) 
    using a Gated Multimodal Unit (GMU).
    """
    # Get parameters from config
    hidden_units = config['model']['hidden_units']
    dropout_rate = config['model']['dropout_rate']
    embed_dim = config['model']['embedding_dim']
    bart_dim = config['model']['bart_dim']
    emo_dim = config['model']['emotion_dim']

    # --- 1. Input Layer Definition ---
    # Semantic Vectors from BART (Summarized review sets)
    user_bart_input = Input(shape=(bart_dim,), dtype=tf.float32, name='user_bart_input')
    item_bart_input = Input(shape=(bart_dim,), dtype=tf.float32, name='item_bart_input')

    # Emotional Vectors from DistilRoBERTa (Averaged individual reviews)
    user_emotion_input = Input(shape=(emo_dim,), dtype=tf.float32, name='user_emotion_input')
    item_emotion_input = Input(shape=(emo_dim,), dtype=tf.float32, name='item_emotion_input')

    # --- 2. Feature Projection ---
    # Project BART vectors to a lower dimension (e.g., 768 -> 256)
    user_bart_proj = bart_projection(user_bart_input, embed_dim, name_prefix='user_bart')
    item_bart_proj = bart_projection(item_bart_input, embed_dim, name_prefix='item_bart')

    # Project Emotion vectors to match the embedding dimension (7 -> 256)
    user_emotion_proj = Dense(embed_dim, activation='tanh', name='user_emotion_proj')(user_emotion_input)
    item_emotion_proj = Dense(embed_dim, activation='tanh', name='item_emotion_proj')(item_emotion_input)

    # --- 3. Feature Fusion via GMU ---
    # Adaptively combine semantic and emotional features for both user and item
    user_feature = GMU(user_bart_proj, user_emotion_proj, units=embed_dim, name_prefix='user')
    item_feature = GMU(item_bart_proj, item_emotion_proj, units=embed_dim, name_prefix='item')

    # --- 4. Rating Prediction (MLP) ---
    # Concatenate the combined user and item latent features
    combined = Concatenate(name='user_item_combined')([user_feature, item_feature])

    # Deep Neural Network for prediction
    x = combined
    for i, units in enumerate(hidden_units):
        x = Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f'dropout_{i+1}')(x)

    # Output layer: Predicting the rating (Linear regression)
    output = Dense(1, activation='linear', name='output_layer')(x)

    # Instantiate the Keras model
    model = Model(
        inputs=[user_bart_input, item_bart_input, user_emotion_input, item_emotion_input], 
        outputs=output
    )

    return model

# -------------------------------------------------------------------------
# Helper Components (Sub-modules)
# -------------------------------------------------------------------------

def GMU(x1, x2, units=256, name_prefix=''):
    """
    Gated Multimodal Unit: 
    Learns an internal weight to adaptively fuse two different modalities.
    """
    # Transformation to a shared latent space
    x1_proj = Dense(units, activation='tanh', name=f'{name_prefix}_proj1')(x1)
    x2_proj = Dense(units, activation='tanh', name=f'{name_prefix}_proj2')(x2)

    # Calculate gate weight (range 0 to 1)
    # The gate decides the importance of semantic vs emotional data
    gate = Dense(units, activation='sigmoid', name=f'{name_prefix}_gate')(
        Concatenate()([x1, x2])
    )

    # Weighted combination: out = gate * x1 + (1 - gate) * x2
    out = Multiply()([gate, x1_proj]) + Multiply()([(1 - gate), x2_proj])
    return out

def bart_projection(x, target_dim=256, name_prefix='bart'):
    """
    Step-wise projection of high-dimensional BART vectors.
    Compresses 768D information into a denser representation.
    """
    x = Dense(512, activation='relu', name=f'{name_prefix}_dense1')(x)
    x = Dense(target_dim, activation='relu', name=f'{name_prefix}_dense2')(x)
    x = Dense(target_dim, activation='tanh', name=f'{name_prefix}_dense3')(x)
    return x