import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, Bidirectional, BatchNormalization
)
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

def create_classification_model(input_shape):
    """
    Builds a classification model using stacked LSTM layers.

    Parameters:
    - input_shape (tuple): Shape of the input data (e.g., (sequence_length, feature_dim)).

    Returns:
    - tf.keras.Model: Compiled Keras model for classification.
    """
    # Input layer
    inputs = Input(shape=input_shape)

    # L2 regularization factor for weights
    l2_reg = regularizers.l2(0.001)

    # 3x LSTM layers
    x = LSTM(
        units=50,
        return_sequences=True,
        kernel_regularizer=l2_reg,
        recurrent_regularizer=l2_reg,
        dropout=0.3
    )(inputs)
    x = BatchNormalization()(x)

    x = LSTM(
        units=50,
        return_sequences=True,
        kernel_regularizer=l2_reg,
        recurrent_regularizer=l2_reg,
        dropout=0.3
    )(x)
    x = BatchNormalization()(x)

    x = LSTM(
        units=50,
        kernel_regularizer=l2_reg,
        recurrent_regularizer=l2_reg,
        dropout=0.3
    )(x)
    x = BatchNormalization()(x)

    # Fully connected layers
    x = Dense(
        units=50,
        activation='relu',
        kernel_regularizer=l2_reg
    )(x)
    x = Dropout(0.3)(x)

    x = Dense(
        units=25,
        activation='relu',
        kernel_regularizer=l2_reg
    )(x)
    x = Dropout(0.3)(x)

    # Output layer for binary classification
    outputs = Dense(
        units=1,
        activation='sigmoid'
    )(x)

    # Define the model
    model = Model(inputs, outputs)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model