# Hand Gesture Recognition System

This project is a Hand Gesture Recognition System that utilizes the **Sign Language MNIST** dataset from [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data). The system uses a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) letters and incorporates a quiz application to test recognition accuracy.

## Dataset

The project uses the **Sign Language MNIST** dataset, which consists of 28x28 grayscale images of ASL hand signs for 26 alphabet letters. You can find the dataset [here](https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data).

## Model Architecture

The model is a CNN built using TensorFlow/Keras with the following layers:
- **Input Layer**: 28x28 grayscale images
- **Conv2D Layer**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer**: 2x2 pooling
- **Conv2D Layer**: 128 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer**: 2x2 pooling
- **Flatten Layer**: Converts multi-dimensional data into a single vector
- **Dense Layer**: 256 neurons, ReLU activation
- **Output Layer**: 26 neurons (one for each letter), Softmax activation

## How to Run

1. **Training the Model**:
   Run the following command to train and save the model:

   ```bash
   python train_and_save_model.py


## Dependencies

This project requires the following Python libraries to run:

- **TensorFlow**: Deep learning framework used for training the sign language recognition model.
- **Keras**: High-level neural network API used to build and train the model.
- **Pandas**: Data manipulation and analysis library used for loading and preprocessing the dataset.
- **NumPy**: Library for numerical computations and matrix operations, utilized for data handling.

### Install all dependencies using:

You can install all the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## sunum.py - Real-Time Sign Language Recognition Game

The `sunum.py` script utilizes the trained sign language model to build an interactive quiz game where users form words by recognizing hand gestures. This real-time application provides a fun and educational experience.

### Features:

- **Real-Time Prediction**: The script captures hand gestures via webcam and predicts corresponding letters using a trained TensorFlow model.
- **Word Formation**: Players complete predefined words by correctly predicting individual letters.
- **Score System**: The score increases with each completed word, and scores are stored in a local SQLite database.
- **Time Limit**: Players must complete as many words as possible within a set time limit.
- **Speech Feedback**: Correctly predicted words are spoken aloud using a text-to-speech engine.

### How the Game Works:

1. **Hand Gesture Detection**: The webcam captures the player's hand gestures, which are processed using MediaPipe to isolate hand landmarks.
2. **Letter Prediction**: The trained model predicts the letter corresponding to the detected hand gesture.
3. **Word Completion**: Players must complete a target word by guessing one letter at a time. If the prediction is correct, the next letter is guessed, and the game continues until the word is complete.
4. **Score and Timer**: A timer limits the gameplay session, and the score increases for each correctly completed word.
5. **Leaderboard**: At the end of the game, scores are saved, and players can view their high scores.

### Running the Game:

To run the game, use the following command:

```bash
python sunum.py
```
### Image examples
![image](https://github.com/user-attachments/assets/38b7c798-1ad9-449b-9cbe-b1bce9ebb061)




