# Import libraries
import pandas as pd
import numpy as np
from skimage import io
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import mode, kurtosis, skew, entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

"""
Features Extraction Function
"""
# Get an image and returns first and second order features of the pixels array 
def get_image_features(image_path):
    # Load the image and converts it into a 2-dimensional numpy array with the pixels values
    image = io.imread(image_path, as_gray = True) * 255
    # Resize image to 512 x 512
    image = resize(image, (512, 512))
    # Converts the float values of the 2-dimensional pixels array into uint8
    image = image.astype(np.uint8)
    
    # Calculate the mean, variance and standard deviation of the 2-dimensional pixels array with numpy functions
    mean = np.mean(image)
    variance = np.var(image)
    std = np.std(image)
    
    # Converts the 2-dimensional pixels array into 1-dimensional
    image_1da = image.flatten()
    
    # Calculate the skewness, kurtosis and entropy of the 1-dimensional array with scipy.stats functions
    skewness = skew(image_1da)
    kurtos = kurtosis(image_1da)
    entro = entropy(image_1da)

    # Calculate the grey-level-co-ocurrence matrix with skimage functions
    # The pixel pair distance offset used is 1
    # The pixel pair angles used are 0, pi/4, pi/2 and 3pi/4
    GLCM = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

    # Calculate texture properties of the grey-level-co-ocurrence matrix 
    contrast = greycoprops(GLCM, 'contrast')[0, 0]
    dissimilarity = greycoprops(GLCM, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(GLCM, 'homogeneity')[0, 0]
    asm = greycoprops(GLCM, 'ASM')[0, 0]
    energy = greycoprops(GLCM, 'energy')[0, 0]
    correlation = greycoprops(GLCM, 'correlation')[0, 0]
    
    # Returns all the features values of the image
    return mean, variance, std, skewness, kurtos, entro, contrast, dissimilarity, homogeneity, asm, energy, correlation

"""
K-Nearest Neighbors Algorithm Functions
"""
# Gets two points and calculate the euclidean distance between them
def euclidean_distance(p1, p2):
    ed = np.sqrt(np.sum((p1 - p2) ** 2))
    return ed

# Function to predict the class with knn model
def knn_predict(x_train, y_train, x_input, n_neighbors):
    # List to store the predictions
    predictions = []
     
    # Loop through the datapoints to be classified
    for i in x_input:   
        # List to store the distances
        distances = []
         
        # Loop through each training data
        for j in range(len(x_train)): 
            # Calculate the euclidean distance
            ed = euclidean_distance(np.array(x_train[j, :]), i) 
            
            # Add the calculated euclidean distance to the list
            distances.append(ed) 
            
        # Convert the list into a numpy array
        distances = np.array(distances) 
         
        # Sort the array while preserving the index
        # Keep the first n_neighbors datapoints
        dist_sorted = np.argsort(distances)[:n_neighbors] 
         
        # Labels of the n_neighbors datapoints from above
        labels = y_train[dist_sorted]
         
        # Determine the majority label in labels
        label = mode(labels).mode[0] 
        predictions.append(label)
        
    # Returns a list with the predictions
    return predictions

"""
Decision Trees Algorithm Functions
"""
def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

"""
Naive Bayes Algorithm Functions
"""
def prior(df, class_column):
    classes = sorted(list(df[class_column].unique()))
    priors = []
    
    for i in classes:
        priors.append(len(df[df[class_column] == i]) / len(df))
    return priors

def likelihood_gaussian(df, feat_name, feat_val, class_column, label):
    feat = list(df.columns)
    df = df[df[class_column] == label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((feat_val - mean) ** 2 / (2 * std ** 2)))
    return p_x_given_y

def nb_predict(df, x_input, class_column):
    features = list(df.columns)[:-1]
    
    priors = prior(df, class_column)
    
    predictions = []
    
    for x in x_input:
        labels = sorted(list(df[class_column].unique()))
        likelihood = [1] * len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= likelihood_gaussian(df, features[i], x[i], class_column, labels[j])
                
        post_prob = [1] * len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * priors[j]
            
        predictions.append(np.argmax(post_prob))
        
    return predictions

dataset = pd.read_csv('data/brain_tumor_dataset.csv', index_col = 0)

dataset = dataset.drop(['image_name', 'label_name'], axis = 1)

train, test = train_test_split(dataset, test_size = 0.2)

x_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

x_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

knn_preds = knn_predict(x_train, y_train, x_train, n_neighbors = 5)
knn_tests = knn_predict(x_train, y_train, x_test, n_neighbors = 5)

nb_preds = nb_predict(train, x_train, class_column = 'label')
nb_tests = nb_predict(train, x_test, class_column = 'label')

dt = DecisionTree(max_depth = 7)
dt.fit(x_train, y_train)
dts_preds = dt.predict(x_train)
dts_tests = dt.predict(x_test)

#print('KNN: %.4f' % (accuracy_score(y_test, knn_tests)))
#print('NB: %.4f' % (accuracy_score(y_test, nb_tests)))
#print('DT: %.4f' % (accuracy_score(y_test, dts_tests)))

meta_model_df = pd.DataFrame(columns = ('knn', 'dts', 'nb', 'true_label'))

meta_model_df['knn'] = knn_preds
meta_model_df['dts'] = dts_preds
meta_model_df['nb'] = nb_preds
meta_model_df['true_label'] = y_train

train_mm, test_mm = train_test_split(meta_model_df, test_size = 0.2)

x_train_mm = train_mm.iloc[:, :-1].values
y_train_mm = train_mm.iloc[:, -1].values

x_test_mm = test_mm.iloc[:, :-1].values
y_test_mm = test_mm.iloc[:, -1].values

stacking_test = pd.DataFrame(columns = ('knn', 'dts', 'nb', 'true_label'))

stacking_test['knn'] = knn_tests
stacking_test['dts'] = dts_tests
stacking_test['nb'] = nb_tests
stacking_test['true_label'] = y_test

x_test_s = stacking_test.iloc[:, :-1].values
y_test_s = stacking_test.iloc[:, -1].values

knn_tests2 = knn_predict(x_train_mm, y_train_mm, x_test_mm, n_neighbors = 5)
s_tests = knn_predict(x_train_mm, y_train_mm, x_test_s, n_neighbors = 5)

#print('Stacking: %.4f' % (accuracy_score(y_test_s, s_tests)))

# Cross Validation
acc_knn = []
acc_nb = []
acc_dts = []
acc_s = []
for i in range(1):
    train, test = train_test_split(dataset, test_size = 0.2)

    x_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values

    x_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values
    
    knn_test = knn_predict(x_train, y_train, x_test, n_neighbors = 5)
    nb_test = nb_predict(train, x_test, class_column = 'label')
    
    dts_test = dt.predict(x_test)
    
    stacking_test = pd.DataFrame(columns = ('knn', 'dts', 'nb', 'true_label'))
    stacking_test['knn'] = knn_test
    stacking_test['dts'] = dts_test
    stacking_test['nb'] = nb_test
    stacking_test['true_label'] = y_test
    
    x_test_s = stacking_test.iloc[:, :-1].values
    
    s_test = knn_predict(x_train_mm, y_train_mm, x_test_s, n_neighbors = 5)

    acc_knn.append(accuracy_score(y_test, knn_test))
    acc_nb.append(accuracy_score(y_test, nb_test))
    acc_dts.append(accuracy_score(y_test, dts_test))
    acc_s.append(accuracy_score(y_test, s_test))
    
print('KNN: %.4f' % (np.mean(acc_knn)))
print('NB: %.4f' % (np.mean(acc_nb)))
print('DT: %.4f' % (np.mean(acc_dts)))
print('Stacking: %.4f' % (np.mean(acc_s)))

print('-------')
print('New prediction')

while True:
    new_image = input('Image path: ')

    features = np.array([get_image_features(new_image)])

    knn_x = knn_predict(x_train, y_train, features, n_neighbors = 5)
    dts_x = dt.predict(features)
    nb_x = nb_predict(train, features, class_column = 'label')

    new_p = np.array([[knn_x[0], dts_x[0], nb_x[0]]])

    s_x = knn_predict(x_train_mm, y_train_mm, new_p, n_neighbors = 5)

    print('No tumor = 0, Glioma tumor = 1, Meningioma tumor = 2, Pituitary tumor = 3')
    print('KNN: '  + str(knn_x[0]))
    print('NB: '  + str(nb_x[0]))
    print('DT: '  + str(dts_x[0]))
    print('Stacking: ' + str(s_x[0]))
    print('\n')


# data/Testing/glioma_tumor/gt_83.jpg
