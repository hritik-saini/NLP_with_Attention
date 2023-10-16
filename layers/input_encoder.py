import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Embedding, LSTM, Dense, TimeDistributed, Concatenate
from tensorflow.keras.models import Model


def input_encoder_fn(input_vocab_size, d_model, n_encoder_layers):
    """ Input encoder runs on the input sentence and creates
    activations that will be the keys and values for attention.

    Args:
        input_vocab_size: int: vocab size of the input
        d_model: int:  depth of embedding (n_units in the LSTM cell)
        n_encoder_layers: int: number of LSTM layers in the encoder
    Returns:
        tf.keras.Model: The input encoder
    """

    # Define input layer
    inputs = Input(shape=(None,))

    # Create an embedding layer to convert tokens to vectors
    embedding = Embedding(input_dim=input_vocab_size, output_dim=d_model)(inputs)

    # Create a list of LSTM layers
    encoder_layers = [LSTM(units=d_model, return_sequences=True) for _ in range(n_encoder_layers)]

    # Apply LSTM layers in sequence
    activations = embedding
    for layer in encoder_layers:
        activations = layer(activations)

    # Create the model
    input_encoder = Model(inputs, activations)

    return input_encoder