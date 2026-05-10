import os
import numpy as np
from keras.models import Sequential
from keras.layers import (Embedding,LSTM,Dense,Dropout)
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

#Preprocessing
texts = []
labels = []
for actor in os.listdir(DATASET):
    actor_path = os.path.join(DATASET, actor)
    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(actor_path, file)
        emotion = file.split("-")[2]
        labels.append(emotion_map[emotion])
        
# as whisper takes a lot of time to transcribe (even when "tiny" is used, hence, the text is extracted from RAVDESS and stored for future runs)
with open(r"Session-4\texts.txt","r",encoding="utf-8") as f:
    for line in f:
        texts.append(line.strip())
        
#tokenisation
word2idx = {}
current_idx = 1
tokenized = []
for text in texts:
    words = text.split()
    seq = []
    for word in words:
        if word not in word2idx:
            word2idx[word] = current_idx
            current_idx += 1
        seq.append(word2idx[word])
    tokenized.append(seq)

#padding 
MAX_LEN = 20
padded = []
for seq in tokenized:
    if len(seq) < MAX_LEN:
        seq = seq + [0] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    padded.append(seq)

#Dataset
X = np.array(padded)
y = to_categorical(labels,num_classes=8)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Text Processing
model = Sequential()
model.add(Embedding(input_dim=len(word2idx) + 1,output_dim=128,input_length=MAX_LEN))
model.add(LSTM(128))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.build(input_shape=(None, MAX_LEN))
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