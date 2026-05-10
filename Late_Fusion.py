import numpy as np
from sklearn.metrics import accuracy_score
from CNN import (
    model as cnn_model,
    X_test as X_audio_test,
    y_test
)
from RNN import (
    model as rnn_model,
    X_test as X_text_test
)

# CNN probabilities
audio_probs = cnn_model.predict(X_audio_test)

# RNN probabilities
text_probs = rnn_model.predict(X_text_test)

# weighted late fusion
final_probs = (0.7 * audio_probs + 0.3 * text_probs)

# final prediction
final_pred = np.argmax(final_probs,axis=1)

# true labels
true_labels = np.argmax(y_test,axis=1)

# accuracy
accuracy = accuracy_score(true_labels,final_pred)
print("\nLate Fusion Accuracy:",accuracy)