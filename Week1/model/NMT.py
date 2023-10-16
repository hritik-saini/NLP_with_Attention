import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Embedding, LSTM, Dense, TimeDistributed, Concatenate
from tensorflow.keras.models import Model

from NLP_With_Attention_coursera.layers.input_encoder import input_encoder_fn
from NLP_With_Attention_coursera.layers.pre_attention_inputs import pre_attention_decoder_fn


def prepare_attention_input(encoder_activations, decoder_activations):
    """Prepare queries, keys, values, and mask for attention.

    Args:
        encoder_activations: tf.Tensor (batch_size, padded_input_length, d_model): output from the input encoder
        decoder_activations: tf.Tensor (batch_size, padded_input_length, d_model): output from the pre-attention decoder
        inputs: tf.Tensor (batch_size, padded_input_length): padded input tokens

    Returns:
        queries, keys, values, and mask for attention.
    """

    # Set the keys and values to the encoder activations
    keys = encoder_activations
    values = encoder_activations

    # Set the queries to the decoder activations
    queries = decoder_activations

    return queries, keys, values


def NMTAttn(input_vocab_size=1000,
            target_vocab_size=1000,
            d_model=1024,
            n_encoder_layers=4,
            n_decoder_layers=4,
            n_attention_heads=8,
            attention_dropout=0.2,
            mode='train'):

    """Returns an LSTM sequence-to-sequence model with attention.

    The input to the model is a pair (input tokens, target tokens), e.g.,
    an English sentence (tokenized) and its translation into German (tokenized).

    Args:
        input_vocab_size: int: vocab size of the input
        target_vocab_size: int: vocab size of the target
        d_model: int:  depth of embedding (n_units in the LSTM cell)
        n_encoder_layers: int: number of LSTM layers in the encoder
        n_decoder_layers: int: number of LSTM layers in the decoder after attention
        n_attention_heads: int: number of attention heads
        attention_dropout: float, dropout for the attention layer
        mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference

    Returns:
        A LSTM sequence-to-sequence model with attention.
    """

    # Step 0: Define input layers
    input_tokens = Input(shape=(None,))
    target_tokens = Input(shape=(None,))

    # Step 1: call the helper function to create layers for the input encoder
    input_encoder = input_encoder_fn(input_vocab_size, d_model, n_encoder_layers)

    # Step 1: call the helper function to create layers for the pre-attention decoder
    pre_attention_decoder = pre_attention_decoder_fn(mode, target_vocab_size, d_model)

    # Step 2: Copy input tokens and target tokens as they will be needed later
    concatenated_tokens = Concatenate(axis=1)([input_tokens, target_tokens])

    # Step 3: Run input encoder on the input and pre-attention decoder on the target
    input_encoder_output = input_encoder(input_tokens[:, :, tf.newaxis])
    pre_attention_decoder_output = pre_attention_decoder(target_tokens[:, :, tf.newaxis])

    # Step 4: Prepare queries, keys, values, and mask for attention
    queries, keys, values = prepare_attention_input(input_encoder_output,
                                                    pre_attention_decoder_output)

    # Step 5: Run the AttentionQKV layer and nest it inside a Residual layer
    attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=n_attention_heads, key_dim=d_model, dropout=attention_dropout)
    attention_output = attention_layer(queries, keys, values) + pre_attention_decoder_output

    # Step 6: Drop attention mask (i.e., index = None)
    # attention_output = attention_output[:, :, :, :d_model]
    # need to add then mask will be added

    # Step 7: Run the rest of the RNN decoder
    lstm_layers = [LSTM(units=d_model, return_sequences=True) for _ in range(n_decoder_layers)]
    decoder_output = attention_output
    for layer in lstm_layers:
        decoder_output = layer(decoder_output)

    # Step 8: Prepare output by making it the right size
    output = Dense(target_vocab_size)(decoder_output)

    # Step 9: Log-softmax for output
    output = tf.math.log(tf.nn.softmax(output))

    # Create the final model
    model = Model(inputs=[input_tokens, target_tokens], outputs=output)

    return model
