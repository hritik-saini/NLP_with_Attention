import tensorflow_datasets as tfds
from NLP_With_Attention_coursera.Helper_Functions import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import os

from NLP_With_Attention_coursera.model.NMT import NMTAttn
from NLP_With_Attention_coursera.preprocessing.dataset import TokenizedDataStream

# Load the TED Talk translation dataset
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

train_tokenizer = TokenizedDataStream(train_examples)
train_stream = train_tokenizer.get_tokenized_stream()

val_tokenizer = TokenizedDataStream(val_examples)
val_stream = val_tokenizer.get_tokenized_stream()

# Define the bound-airs and batch_sizes for the bucketing
boundaries = [8, 16, 32, 64, 128, 256, 512]
batch_sizes = [256, 128, 64, 32, 16, 8, 4, 2]

input_vocab_size = train_tokenizer.pt_tokenizer.num_words
target_vocab_size = train_tokenizer.en_tokenizer.num_words
model = NMTAttn(input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size, d_model=100, mode='train')
print(model.summary())

# Define the output directory
output_dir = 'output_dir/'

# Remove old model if it exists to restart training


# Create the generators.
train_batch_stream = Bucket_By_Length(
    boundaries, batch_sizes,
    length_keys=[0, 1]  # As before: count inputs and targets to length.
)(train_stream)

eval_batch_stream = Bucket_By_Length(
    boundaries, batch_sizes,
    length_keys=[0, 1]  # As before: count inputs and targets to length.
)(val_stream)

train_data_list = []
eval_data_list = []

# Load train data
for data_point in train_batch_stream:
    input_data, target_data = data_point
    train_data_list.append((input_data, target_data))

# Load eval data
for data_point in eval_batch_stream:
    input_data, target_data = data_point
    eval_data_list.append((input_data, target_data))


# Define the training and evaluation data generators
def data_generator(data):
    for input_data, target_data in data:
        yield [input_data, target_data], target_data


# Define the loss function
loss_fn = SparseCategoricalCrossentropy()

# Define the optimizer with the specified learning rate
optimizer = Adam(learning_rate=0.01)

# Define a metric for accuracy
accuracy_metric = SparseCategoricalAccuracy()

# Compile the model with the optimizer, loss, and metric
model.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy_metric])


if os.path.exists(output_dir + 'model.h5'):
    os.remove(output_dir + 'model.h5')

# Define the number of training epochs
num_epochs = 500  # You can adjust this as needed

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    train_data_generator = data_generator(train_data_list)
    eval_data_generator = data_generator(eval_data_list)

    # # Training step
    history = model.fit(
        train_data_generator,
        epochs=1,  # One epoch at a time,
        verbose=1  # Set to 1 for progress updates
    )

    # Extract training loss and accuracy from the history object
    train_loss = history.history['loss'][0]
    train_accuracy = history.history['sparse_categorical_accuracy'][0]

    # Validation step
    eval_loss, eval_accuracy = model.evaluate(eval_data_generator, verbose=1)

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        # Print training and validation metrics
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {eval_loss:.4f}, Validation Accuracy: {eval_accuracy:.4f}")

        # Save the model checkpoint after each epoch
        model.save(output_dir + f"model_epoch_{epoch + 1}.h5")
