import numpy as np

def encode(y):
    unique = np.unique(y)
    map = {label: idx for idx, label in enumerate(unique)}
    encoded = np.array([map[label] for label in y])
    return encoded, map

def decode(y_en, map):
    rev = {v: k for k, v in map.items()}
    return np.array([rev[val] for val in y_en])

def gini(y):
    prob=np.bincount(y)/len(y)
    return 1 - np.sum(prob**2)

# BONUS
def entropy(y):
    counts = np.bincount(y)
    prob=counts/len(y)
    prob=prob[prob > 0]
    return -np.sum(prob*np.log2(prob))
def best_split_entropy(X, y):
    best_entropy=float('inf')
    best_feature=None
    best_threshold=None

    for i in range(X.shape[1]):
        val=X[:,i]
        uniq=np.unique(val)

        for j in range(len(uniq)-1):
            thres=(uniq[j]+uniq[j+1])/2

            left_mask=val<=thres
            right_mask=val>thres

            y_left=y[left_mask]
            y_right=y[right_mask]

            if len(y_left)==0 or len(y_right)==0:
                continue

            n=len(y)
            w_entropy=(len(y_left)/n)*entropy(y_left)+(len(y_right)/n)*entropy(y_right)

            if w_entropy<best_entropy:
                best_entropy=w_entropy
                best_feature=i
                best_threshold=thres

    return best_feature, best_threshold

def best_split(X,y_en):
    best_gini = float('inf')
    best_feature = None
    best_thres = None
    for i in range(X.shape[1]):
        val=X[:,i]
        
        uniq=np.unique(val)
        for j in range(len(uniq)-1):
            thres=(uniq[j]+uniq[j+1])/2
            
            left_i=val<=thres
            right_i=val>thres
            
            left_y=y_en[left_i]
            right_y=y_en[right_i]

            if len(left_y) == 0 or len(right_y) == 0:continue
            n = len(y)
            w_gini = (len(left_y)/n)*gini(left_y) + (len(right_y)/n)*gini(right_y)
            
            if(w_gini<best_gini):
                best_gini = w_gini
                best_feature = i
                best_thres = thres
                
    return best_feature,best_thres

data = [
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
]
d=np.array(data)

X = np.array([row[:3] for row in data],dtype=float)
y = np.array([row[3] for row in data])

y_en,map=encode(y)
best_f,best_t=best_split(X,y_en)

class Node:
    def __init__(self,feature_index=None,thres=None,left=None,right=None,value=None):
        self.feature_index=feature_index
        self.thres=thres       
        self.left=left                   
        self.right=right                 
        self.value=value              
    
def tree(X, y, depth=0, max_depth=3, min_samples=1):
    if len(np.unique(y))==1:
        return Node(value=y[0])

    if depth>=max_depth or len(y)<min_samples:
        val=np.bincount(y).argmax()
        return Node(value=val)

    feature,thres=best_split(X, y)

    if feature is None:
        val=np.bincount(y).argmax()
        return Node(value=val)

    val=X[:, feature]
    left_i=val<=thres
    right_i=val>thres

    X_left,y_left=X[left_i],y[left_i]
    X_right,y_right=X[right_i],y[right_i]

    lc=tree(X_left,y_left,depth+1,max_depth,min_samples)
    rc=tree(X_right,y_right,depth+1,max_depth,min_samples)

    return Node(feature,thres,lc,rc)

def predict_one(x,node):
    if node.value!=None:
        return node.value

    if x[node.feature_index]<=node.thres:
        return predict_one(x,node.left)
    else:
        return predict_one(x,node.right)

def predict(X,tree):
    return np.array([predict_one(x,tree) for x in X])
    
def accuracy(pred,y_en):
    return np.mean(y_en==pred);

test_data = np.array([
    [6.0, 2.1, 0],   # Expected: Beer
    [39.0, 0.05, 1], # Expected: Whiskey
    [13.0, 1.3, 1]   # Expected: Wine
])
y_exp=['Beer','Whiskey','Wine']

d_tree=tree(X,y_en)
preds=predict(test_data,d_tree)
final_preds=decode(preds,map)

print(final_preds)
print(accuracy(final_preds,y_exp))