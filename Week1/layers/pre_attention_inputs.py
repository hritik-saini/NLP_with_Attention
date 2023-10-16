import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Embedding, LSTM, Dense, TimeDistributed, Concatenate
from tensorflow.keras.models import Model


def pre_attention_decoder_fn(mode, target_vocab_size, d_model):
    """ Pre-attention decoder runs on the targets and creates
    activations that are used as queries in attention.

    Args:
        mode: str: 'train' or 'eval'
        target_vocab_size: int: vocab size of the target
        d_model: int:  depth of embedding (n_units in the LSTM cell)
    Returns:
        tf.keras.Model: The pre-attention decoder
    """

    # Define input layer
    inputs = Input(shape=(None,))

    # Shift right to insert start-of-sentence token and implement teacher forcing during training
    # shifted_right = ShiftRightLayer(mode=mode,n_positions=1)(inputs) # This will shift the right to insert the start of sentence token and implement teacher forcing during training

    batch_size = tf.shape(inputs)[0]
    zero_padding = tf.zeros((batch_size, 1), dtype=tf.float32)
    # print(zero_padding)
    shifted_right = tf.concat([zero_padding, inputs[:, :-1]], axis=1)

    # Create an embedding layer to convert tokens to vectors
    embedding = Embedding(input_dim=target_vocab_size, output_dim=d_model)(shifted_right)

    # Create an LSTM layer
    lstm = LSTM(units=d_model, return_sequences=True)(embedding)

    # Create the model
    pre_attention_decoder = Model(inputs, lstm)

    return pre_attention_decoder

