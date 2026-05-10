from keras.models import Model,Sequential
from keras.layers import (
    Dense,
    Dropout,
    Concatenate
)
from CNN import (
    model as cnn_model,
    X_test as X_audio_test,
    X_train as X_audio_train,
    y_train,
    y_test
)
from RNN import (
    model as rnn_model,
    X_test as X_text_test,
    X_train as X_text_train
)
# remove final softmax layers
cnn_feature_extractor = Model(inputs=cnn_model.inputs,outputs=cnn_model.layers[-2].output)
rnn_feature_extractor = Model(inputs=rnn_model.inputs,outputs=rnn_model.layers[-2].output)

# extract features
cnn_train_features = (cnn_feature_extractor.predict(X_audio_train))
cnn_test_features = (cnn_feature_extractor.predict(X_audio_test))

rnn_train_features = (rnn_feature_extractor.predict(X_text_train))
rnn_test_features = (rnn_feature_extractor.predict(X_text_test))

# concatenate features
X_train_fusion = Concatenate()([cnn_train_features,rnn_train_features])
X_test_fusion = Concatenate()([cnn_test_features,rnn_test_features])

# fusion classifier
fusion_input_shape = (X_train_fusion.shape[1],)

fusion_model = Sequential()
fusion_model.add(Dense(64,activation='relu',input_shape=fusion_input_shape))
fusion_model.add(Dropout(0.3))
fusion_model.add(Dense(8,activation='softmax'))

fusion_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
fusion_model.summary()

#Training
fusion_model.fit(
    X_train_fusion,
    y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.1
)

loss, accuracy = fusion_model.evaluate(X_test_fusion,y_test)

print("\nEarly Fusion Accuracy:",accuracy)