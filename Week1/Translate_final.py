import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds

from NLP_With_Attention_coursera.preprocessing.dataset import TokenizedDataStream


def sampling_decode(input_sentence, NMTAttn_model):
    """Returns the translated sentence.

    Args:
        input_sentence (str): sentence to translate.
        NMTAttn_model (tf.keras.Model): An LSTM sequence-to-sequence model with attention.
        temperature (float): parameter for sampling ranging from 0.0 to 1.0.
            0.0: same as argmax, always pick the most probable token
            1.0: sampling from the distribution (can sometimes say random things)

    Returns:
        tuple: (list, str)
            list of int: tokenized version of the translated sentence
            str: the translated sentence
    """

    # Tokenize the input sentence using the tokenizer used during training
    input_tokens = train_tokenizer.pt_tokenizer.texts_to_sequences([input_sentence])[0]

    input_tokens = [input_tokens]

    print(input_tokens)

    # Initialize the list of output tokens
    cur_output_tokens = []

    # Initialize an integer that represents the current output index
    cur_output = 0

    # Set the encoding of the "end of sentence" as 1
    EOS = 1

    # Check that the current output is not the end of sentence token
    while len(cur_output_tokens) != 4:
        # Update the current output token by getting the index of the next word (hint: use next_symbol)
        cur_output, _ = next_symbol(NMTAttn_model, input_tokens, cur_output_tokens)

        # Append the current output token to the list of output tokens
        cur_output_tokens.append(cur_output)

    # Detokenize the output tokens using the tokenizer used during training
    sentence = train_tokenizer.detokenize(cur_output_tokens, "Target")

    return cur_output_tokens, sentence


def next_symbol(NMTAttn_model, input_tokens, cur_output_tokens):
    """Returns the index of the next token.

    Args:
        NMTAttn (tf.keras.Model): An LSTM sequence-to-sequence model with attention.
        input_tokens (tf.Tensor): tokenized representation of the input sentence
        cur_output_tokens (list): tokenized representation of previously translated words

    Returns:
        int: index of the next token in the translated sentence
        float: log probability of the next symbol
    """

    # Set the length of the current output tokens
    token_length = len(cur_output_tokens)

    # Calculate the next power of 2 for padding length
    padded_length = 2 ** int(np.ceil(np.log2(token_length + 1)))

    # Pad cur_output_tokens up to the padded_length
    padded = cur_output_tokens + [0] * (padded_length - token_length)

    # Model expects the output to have an axis for the batch size in front,
    # so convert `padded` list to a TensorFlow tensor with shape (1, <padded_length>)
    padded_with_batch = tf.constant([padded])

    input_tokens = tf.convert_to_tensor(input_tokens)
    input_given = [input_tokens, padded_with_batch]
    # Get the model prediction (using the `NMTAttn` model)
    output = NMTAttn_model(inputs=input_given)
    print(output)

    # Get log probabilities from the last token output
    log_probs = output[0, token_length, :]

    # Get the next symbol by getting a logsoftmax sample
    symbol = tf.squeeze(tf.random.categorical([log_probs], 1))

    return symbol.numpy(), float(log_probs[symbol])


if __name__ == '__main__':

    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    train_tokenizer = TokenizedDataStream(train_examples)
    # Load the saved model
    # Clear the TensorFlow session
    model_path = 'output_dir/model_epoch_1.h5'  # Replace <epoch_number> with the desired epoch number
    loaded_model = load_model(model_path, compile=False)
    result_tokens, translated_sentence = sampling_decode("eu te amo", loaded_model)
    print("Translated tokens:", result_tokens)
    print("Translated sentence:", translated_sentence)
