import numpy as np
import matplotlib.pyplot as plt

# Label Encoding

def encode(y):
    unique = np.unique(y)
    map = {label: idx for idx, label in enumerate(unique)}
    encoded = np.array([map[label] for label in y])
    return encoded, map

def decode(y_en, map):
    rev = {v: k for k, v in map.items()}
    return np.array([rev[val] for val in y_en])

# Train-Test Split

def train_test_split(X,y,size=0.3):
    n=len(X)
    split=int(n*(1-size))
    X_train=X[:split]
    X_test=X[split:]
    y_train=y[:split]
    y_test=y[split:]
    
    return X_train,X_test,y_train,y_test

# Distance Functions

# Euclidean

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Manhattan 

def man_dist(a,b):
    return np.sum(np.abs(a-b))

# Minkowski

def min_dist(a,b,p):
    return (np.sum(np.abs(a-b)**p))**(1/p)

# KNN Classifier

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict_one(self, x):
        distances = []

        for i in range(len(self.X_train)):
            d = euclidean_distance(x, self.X_train[i])
            distances.append((d, self.y_train[i]))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]

        labels = [label for _, label in neighbors]
        unique, counts = np.unique(labels, return_counts=True)
        return unique[np.argmax(counts)]

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []

        for x in X_test:
            predictions.append(self.predict_one(x))

        return np.array(predictions)

# Accuracy Function

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Test Section

data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
]

data = np.array(data, dtype=object)

# Features
X = data[:, :3].astype(float)

# Labels
y_str = data[:, 3]
y, map = encode(y_str)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, size=0.3)

# Model
model = KNN(k=3)
model.fit(X_train, y_train)

# Predictions(encoded)
preds_encoded = model.predict(X_test)

# Decode predictions
preds = decode(preds_encoded, map)
y_test_decoded = decode(y_test, map)

print("Predictions:", preds)
print("Actual:", y_test_decoded)
print("Accuracy:", accuracy(y_test, preds_encoded))

#2-D Plotter -> Plots 2 features

def plot_boundary(model, X, y, resolution=200):
    X = np.array(X)
    y = np.array(y)
    X_plot=X[:,:2]
    x_min,x_max=X_plot[:,0].min()-1,X_plot[:,0].max()+1
    y_min,y_max=X_plot[:,1].min()-1,X_plot[:,1].max()+1

    xx,yy=np.meshgrid(
        np.linspace(x_min,x_max,resolution),
        np.linspace(y_min,y_max,resolution)
    )

    grid=np.c_[xx.ravel(),yy.ravel()]
    if X.shape[1]>2:
        grid_full=np.zeros((grid.shape[0],X.shape[1]))
        grid_full[:,:2] =grid
    else:
        grid_full=grid

    Z=model.predict(grid_full)
    Z=np.array(Z).reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy,Z,alpha=0.3)
    plt.scatter(X_plot[:,0],X_plot[:,1],c=y,edgecolor='k')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"KNN Decision Boundary (k={model.k})")

    plt.show()

plot_boundary(model,X,y)
