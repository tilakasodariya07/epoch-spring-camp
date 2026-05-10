# Speech Emotion Recognition using CNN, RNN and Multimodal Fusion

## Objective

The objective of this project is to classify emotions from speech using:

* Audio based CNN
* Text based RNN
* Multimodal Fusion

The dataset used is RAVDESS.

# Dataset

The RAVDESS dataset contains speech recordings labelled with 8 emotions.

| Emotion ID | Emotion   |
| ---------- | --------- |
| 01         | Neutral   |
| 02         | Calm      |
| 03         | Happy     |
| 04         | Sad       |
| 05         | Angry     |
| 06         | Fearful   |
| 07         | Disgust   |
| 08         | Surprised |

# Audio Preprocessing

Audio files were loaded using Librosa.
Steps performed:

* Load audio
* Convert to Mel Spectrogram
* Convert power spectrogram to decibel scale
* Normalize spectrogram
* Padding / truncation
* Reshape for CNN input

Mel Spectrograms were used because they represent frequency and time information effectively.

# Text Preprocessing

Speech clips were transcribed using OpenAI Whisper.
Steps performed:

* Speech to text transcription
* Tokenization
* Integer encoding
* Padding

The padded sequences were then used as input to the RNN.

# CNN Architecture

Architecture:

Input Spectrogram
* Conv2D(32)
* MaxPooling2D
* Conv2D(64)
* MaxPooling2D
* Conv2D(128)
* MaxPooling2D
* Flatten
* Dense(64)
* Dropout(0.3)
* Softmax Output

The convolution layers learn patterns related to:
* pitch
* intensity
* energy variations
* temporal frequency relationships

MaxPooling reduces dimensions while preserving dominant features.
Dropout was added to reduce overfitting.

# RNN Architecture

Architecture:

Text Input
* Embedding Layer
* LSTM(128)
* Dropout(0.3)
* Dense(64)
* Softmax Output

LSTMs are suitable for sequential text data because they can retain contextual information across tokens.
The embedding layer converts words into dense vector representations.

# Early Fusion Architecture

Feature vectors from the CNN and RNN bottleneck layers were concatenated and passed through dense layers.

Architecture:

Audio CNN Features
                   \
                    \
                      → Concatenate → Dense → Softmax
                    /
                   /
Text RNN Features

Early Fusion allows the model to learn relationships between:
* audio features
* text features

during training itself.

# Late Fusion Architecture

Late Fusion combines prediction probabilities from separately trained CNN and RNN models.
Weighted averaging was used:-> Final Probability = 0.7 × Audio Probability + 0.3 × Text Probability

as the CNN performed much better than the RNN,therefore, greater importance was assigned to the audio modality.


# Loss Function

Categorical Cross Entropy was used because this is a multi-class classification problem.

Optimizer used:
* Adam

Metrics used:
* Accuracy
* Precision
* Recall
* F1 Score

# Results

| Model        | Accuracy |
| ------------ | -------- |
| Audio CNN    | 61.46%   |
| Text RNN     | 12.85%   |
| Early Fusion | 61.81%   |
| Late Fusion  | 61.81%   |

# Accuracy Comparison

The following graph compares the accuracy of:

* CNN
* RNN
* Early Fusion
* Late Fusion

[Accuracy Comparison](<Accuracy_Comp.png>)

---

# Confusion Matrix

The confusion matrix for Late Fusion shows the distribution of predicted and actual emotions.

It helps visualize:

* commonly confused emotions
* class-wise strengths and weaknesses
* prediction distribution

[Confusion Matrix](<Conf_Matrix.png>)

# Analysis

## CNN
The CNN achieved strong performance because emotional information in RAVDESS is mainly present in:
* tone
* pitch
* speaking intensity
* energy variations

## RNN
The RNN achieved low standalone accuracy because the RAVDESS transcripts contain repetitive sentence structures with limited emotional semantic variation.

## Early Fusion
Early Fusion slightly outperformed the standalone CNN model.
Although the text modality individually performed poorly, combining feature representations from both modalities provided complementary information.


{
  Early Fusion Classification Report:

                  precision    recall  f1-score

        neutral       0.29      0.30      0.29 
           calm       0.62      0.59      0.60 
          happy       0.50      0.38      0.43 
            sad       0.35      0.38      0.37 
          angry       0.71      0.71      0.71 
        fearful       0.51      0.66      0.58 
        disgust       0.56      0.56      0.56 
      surprised       0.76      0.69      0.72 

       accuracy                           0.56 
      macro avg       0.54      0.53      0.53 
   weighted avg       0.56      0.56      0.56 

}

## Late Fusion
Late Fusion achieved performance comparable to Early Fusion.
Weighted averaging allowed the stronger audio modality to dominate predictions while still incorporating information from the text modality.

{
  Late Fusion Classification Report:

                  precision    recall  f1-score

        neutral       0.14      0.10      0.12 
           calm       0.77      0.61      0.68 
          happy       0.54      0.21      0.30 
            sad       0.37      0.56      0.44 
          angry       0.77      0.71      0.74 
        fearful       0.66      0.72      0.69 
        disgust       0.54      0.66      0.59 
      surprised       0.62      0.73      0.67 

       accuracy                           0.57 
      macro avg       0.55      0.54      0.53 
   weighted avg       0.58      0.57      0.56 

}

# Training and Validation Loss
Training and validation loss plots were generated for:
* CNN
* RNN
* Early Fusion
These plots helped analyze:
* convergence
* overfitting
* generalization performance

# Confusion Analysis
Some common confusions observed:
* calm and neutral
* happy and surprised
* fearful and surprised

The model performed strongest on:
* angry
* calm
* surprised

while performance on:
* happy
* sad
was comparatively weaker.

# Libraries and Tools

| Component        | Library             |
| ---------------- | ------------------- |
| Deep Learning    | TensorFlow / Keras  |
| Audio Processing | Librosa             |
| Speech-to-Text   | OpenAI Whisper      |
| Evaluation       | Scikit-learn        |
| Visualization    | Matplotlib, Seaborn |

# Conclusion
The project demonstrated that audio features are significantly more informative than transcript semantics for emotion recognition in the RAVDESS dataset.

Although the text-only RNN performed poorly, multimodal fusion slightly improved overall performance over the standalone CNN model.

Both Early Fusion and Late Fusion achieved comparable performance, showing that even weak modalities can provide complementary information when integrated properly.
