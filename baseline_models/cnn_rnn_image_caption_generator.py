# --- This script generates the model diagram for your documentation ---
# --- It does NOT generate captions ---

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, add
from tensorflow.keras.utils import plot_model
import os

# --- This is the model definition you provided ---
def define_cnn_rnn_model(vocab_size, max_length):
    """
    Defines the classic CNN-LSTM 'merge' architecture.
    """
    # CNN (Encoder) Part
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # RNN (Decoder) Part
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Merge (Decoder) Part
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Build the Model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# --- Example Usage ---
if __name__ == "__main__":
    # You will need tensorflow, pydot, and graphviz to generate the plot
    # 1. pip install tensorflow pydot
    # 2. On Mac: brew install graphviz
    
    # Define some example parameters
    VOCABULARY_SIZE = 10000  # Example: 10k unique words
    MAX_CAPTION_LENGTH = 34  # Example: 34 words max
    
    print("--- Generating CNN-RNN Baseline Model Diagram ---")
    
    try:
        baseline_model = define_cnn_rnn_model(VOCABULARY_SIZE, MAX_CAPTION_LENGTH)
        
        # Save the model summary to a text file
        summary_path = 'cnn_rnn_summary.txt'
        with open(summary_path, 'w') as f:
            baseline_model.summary(print_fn=lambda x: f.write(x + '\n'))
            
        # Save the model plot to an image file
        plot_path = 'cnn_rnn_baseline_model.png'
        plot_model(baseline_model, to_file=plot_path, show_shapes=True)
        
        print(f"\nSuccessfully saved diagram and summary:")
        print(f"- {os.path.abspath(plot_path)}")
        print(f"- {os.path.abspath(summary_path)}")

    except Exception as e:
        print(f"\n--- Error ---")
        print(f"Could not generate model diagram. Error: {e}")
        print("Please make sure you have installed 'tensorflow', 'pydot', and 'graphviz'.")
        print("Run: pip install tensorflow pydot")
        print("On Mac, also run: brew install graphviz")