# Trigger Word Detection

This project implements a deep learning model to detect a specific "trigger word" or "wake word" in a 10-second audio clip. When the trigger word "activate" is detected, the model outputs a signal, which is used here to overlay a chime sound on the original audio. This is similar to the technology used in voice assistants like "Hey Siri" or "Ok Google".

The model uses a combination of a 1D Convolutional Neural Network (CNN) and Gated Recurrent Units (GRUs) to analyze the spectrogram of the audio data and make predictions at each time step.

## 🚀 Project Pipeline

The implementation follows these key steps:

1.  **Data Synthesis**: Since a large, labeled dataset of trigger words is often unavailable, this project synthesizes its own training data. It takes 10-second background audio clips and overlays them with random occurrences of the "activate" word (positives) and other random words (negatives).

2.  **Label Generation**: For each synthesized audio clip, a corresponding label vector is created. A `1` is placed in the label vector for a short duration immediately after the "activate" word ends, and `0` everywhere else.

3.  **Audio to Spectrogram**: Raw audio clips are converted into spectrograms, which serve as the input features for the model. A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time.

4.  **Model Architecture**: The model is built using Keras and consists of:
    *   A 1D Convolutional layer to extract features from the spectrogram.
    *   Two stacked GRU layers to process the sequential nature of the audio data.
    *   A `TimeDistributed` Dense layer with a sigmoid activation function to output a detection probability for each time step.

5.  **Training**: The project demonstrates fine-tuning a pre-trained model on the newly synthesized dataset.

6.  **Inference & Prediction**: The trained model is used to predict the presence of the trigger word in a new audio file. If the output probability remains above a certain threshold for a specified number of consecutive time steps, a chime sound is overlaid at that point in the audio.

## 📁 Repository Structure

.
├── audio_examples/       # Contains example audio files for testing
├── models/               # Stores the pre-trained model architecture and weights
├── raw_data/             # Contains raw audio clips (activates, negatives, backgrounds)
├── README.md             # This file
├── Trigger_word_detection.ipynb # Jupyter Notebook with the full implementation
├── chime_output.wav      # Example output file after detection
├── insert_test.wav       # Intermediate file used for testing data synthesis
├── td_utils.py           # Utility functions for the project
├── tmp.wav               # Temporary file created during prediction
├── train.py              # Script for training the model
└── train.wav             # An example of a synthesized training audio file

## 🛠️ How to Use

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    Ensure you have the required libraries installed.
    ```bash
    pip install numpy pydub tensorflow jupyter matplotlib scipy
    ```

3.  **Run the Jupyter Notebook:**
    Launch Jupyter Notebook and open `Trigger_word_detection.ipynb`.
    ```bash
    jupyter notebook
    ```

4.  **Test with your own audio:**
    *   Place your `.wav` audio file in the `audio_examples/` directory.
    *   Navigate to the last section of the notebook ("Test on your own audio").
    *   Change the `your_filename` variable to point to your audio file.
    *   Run the cells to preprocess your audio, run detection, and listen to the output with the chime.

## ⚙️ Dependencies

*   Python 3.x
*   TensorFlow / Keras
*   NumPy
*   SciPy
*   pydub
*   Matplotlib
*   Jupyter Notebook
*   IPython
