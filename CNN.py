import os
import librosa as li
import numpy as np
from keras.models import Sequential
from keras.layers import (Conv2D,MaxPooling2D,Flatten,Dense,Dropout)
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATASET = "Session-4/DATA"
emotion_map = {
    "01": 0,
    "02": 1,
    "03": 2,
    "04": 3,
    "05": 4,
    "06": 5,
    "07": 6,
    "08": 7
}

# Preprocessing
spectrograms = []
labels = []
for actor in os.listdir(DATASET):
    actor_path = os.path.join(DATASET, actor)
    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(actor_path, file)
        y, sr = li.load(path,sr=22050)
        mel = li.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        mel_db = li.power_to_db(mel,ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / mel_db.std() # standardize the spectograms;
        # Reshaping for CNN Input
        MAX_LEN = 150
        if mel_db.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mel_db.shape[1]
            mel_db = np.pad(mel_db,((0,0),(0,pad_width)),mode='constant')
        else:
            mel_db = mel_db[:, :MAX_LEN]
        mel_db = np.expand_dims(
            mel_db,
            axis=-1
        )
        spectrograms.append(mel_db)
        emotion = file.split("-")[2]
        labels.append(emotion_map[emotion])

# Dataset
X = np.array(spectrograms)
y = to_categorical(labels,num_classes=8)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Audio Processing
model = Sequential()

#Series of Converging and Pooling to make the model learn more characheristic features of the audio
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,150,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))

#MaxPooling extracts the dominant feature of the matrix
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3)) #to deacrease the overfitting
model.add(Dense(8,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.build(input_shape=(None, 128, 150, 1))
model.summary()

#Training
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.1
)
model.predict(X_train[:1])
loss, accuracy = model.evaluate(X_test,y_test)
print("\nTest Accuracy:", accuracy)