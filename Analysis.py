import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from CNN import (
    model as cnn_model,
    X_test as X_audio_test,
    y_test
)
from RNN import (
    model as rnn_model,
    X_test as X_text_test
)
from Early_Fusion import (
    fusion_model as early_fusion_model,
    X_test_fusion
)
from Late_Fusion import (
    final_probs as late_fusion_probs
)
emotion_names = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]
# true labels
true_labels = np.argmax(y_test,axis=1)

# CNN
cnn_probs = cnn_model.predict(X_audio_test)
cnn_pred = np.argmax(cnn_probs,axis=1)
cnn_accuracy = accuracy_score(true_labels,cnn_pred)
print("\nCNN Accuracy:", cnn_accuracy)
print("\nCNN Classification Report:\n")
print(classification_report(true_labels,cnn_pred,target_names=emotion_names))

# RNN
rnn_probs = rnn_model.predict(X_text_test)
rnn_pred = np.argmax(rnn_probs,axis=1)
rnn_accuracy = accuracy_score(true_labels,rnn_pred)
print("\nRNN Accuracy:", rnn_accuracy)
print("\nRNN Classification Report:\n")
print(classification_report(true_labels,rnn_pred,target_names=emotion_names))

# EARLY FUSION
early_probs = early_fusion_model.predict(X_test_fusion)
early_pred = np.argmax(early_probs,axis=1)
early_accuracy = accuracy_score(true_labels,early_pred)
print("\nEarly Fusion Accuracy:",early_accuracy)
print("\nEarly Fusion Classification Report:\n")
print(classification_report(true_labels,early_pred,target_names=emotion_names))

# LATE FUSION
late_pred = np.argmax(late_fusion_probs,axis=1)
late_accuracy = accuracy_score(true_labels,late_pred)
print("\nLate Fusion Accuracy:",late_accuracy)
print("\nLate Fusion Classification Report:\n")
print(classification_report(true_labels,late_pred,target_names=emotion_names))

# ACCURACY BAR GRAPH
models = ["CNN","RNN","Early Fusion","Late Fusion"]
accuracies = [cnn_accuracy,rnn_accuracy,early_accuracy,late_accuracy]
plt.figure(figsize=(8,5))
plt.bar(models,accuracies)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()

# CONFUSION MATRIX

cm = confusion_matrix(true_labels,late_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=emotion_names,yticklabels=emotion_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Late Fusion Confusion Matrix")
plt.show()