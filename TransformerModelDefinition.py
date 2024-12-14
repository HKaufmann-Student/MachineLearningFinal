# TransformerModelDefinition.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D
)
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.3, l2_reg=0.001):
    """
    Defines a Transformer encoder block with multi-head attention and feed-forward layers.

    Parameters:
    - inputs (tf.Tensor): Input tensor for the encoder.
    - head_size (int): Dimensionality of each attention head.
    - num_heads (int): Number of attention heads.
    - ff_dim (int): Dimensionality of the feed-forward layer.
    - dropout (float): Dropout rate for regularization.
    - l2_reg (float): L2 regularization factor for weights.

    Returns:
    - tf.Tensor: Output tensor of the encoder block.
    """
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout,
        kernel_regularizer=regularizers.l2(l2_reg),
        bias_regularizer=regularizers.l2(l2_reg)
    )(x, x)
    x = Dropout(dropout)(x)
    res = tf.keras.layers.Add()([x, inputs])

    # Feed-forward network
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1], kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout)(x)
    return tf.keras.layers.Add()([x, res])

def create_classification_model(input_shape):
    """
    Builds a classification model using stacked Transformer encoder blocks.

    Parameters:
    - input_shape (tuple): Shape of the input data (e.g., (sequence_length, feature_dim)).

    Returns:
    - tf.keras.Model: Compiled Keras model for classification.
    """
    inputs = Input(shape=input_shape)

    # L2 regularization factor for weights
    l2_reg = 0.001

    # Add Transformer encoder blocks
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.3, l2_reg=l2_reg)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.3, l2_reg=l2_reg)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.3, l2_reg=l2_reg)

    # Global average pooling to reduce sequence dimension
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)

    # Dense (fully connected) layers
    x = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(0.3)(x)

    x = Dense(25, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(0.3)(x)

    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(x)

    # Define the model
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
