#!/usr/bin/env python3
# ------------------------- CODE CELL injected: setup --------------------------
__IB_FLAG__ = True
__IMPORT_FILTER__ = globals().get('IMPORT_FILTER', None)
__PLOT_REGISTRY__ = []
__LABEL__ = None


# If matplotlib is available on the test system, it's set to headless mode and all plots are stored
# on disk rather than displayed.
try:
    print("use 'Agg' backend for matplotlib")
    import matplotlib as mpl
    mpl.use('Agg')

    import matplotlib.pyplot as plt

    # store current matplotlib figure to disk
    def _dump_figure():
        from pathlib import Path
        from autograde.util import snake_case

        # check if something was plotted
        if plt.gcf().get_axes():
            global __LABEL__
            global __PLOT_REGISTRY__

            # ensure plots folder exists
            root = Path('figures')
            if not root.is_dir():
                root.mkdir()

            # infer name
            name = snake_case(f'fig_{__LABEL__}_{len(__PLOT_REGISTRY__) + 1}')
            path = root.joinpath(name)
            __PLOT_REGISTRY__.append(path)

            # store current figure
            print(f'save current figure: {path}')
            plt.savefig(path)
            plt.close()

    plt.show = lambda *args, **kwargs: _dump_figure()

except ImportError:
    pass


dump_figure = globals().get('_dump_figure', lambda *args, **kwargs: None)


# inform user about import filters used
if __IMPORT_FILTER__ is not None:
    regex, blacklist = __IMPORT_FILTER__
    print(f'set import filter: regex=r"{regex}", blacklist={blacklist}')

# EXECUTED IN 0.121s
# STDOUT
#     use 'Agg' backend for matplotlib


# ------------------------------- CODE CELL nb-1 -------------------------------
# credentials of all team members (you may add or remove items from the dictionary, but keep this dictionary structure)
team_members = [
    {
        'first_name': 'Na Young',
        'last_name': 'Ahn',
        'student_id': 392326
    },
    {
        'first_name': 'Esther',
        'last_name': 'Tala',
        'student_id': 368095
    },
    {
        'first_name': 'Mika',
        'last_name': 'Rosin',
        'student_id': 395049
    },
    {
        'first_name': 'Laurin',
        'last_name': 'Ellenbeck',
        'student_id': 372280
    }
]

# injected by test
dump_figure()

# EXECUTED IN 0.00149s


# ------------------------------- CODE CELL nb-2 -------------------------------
# general imports may go here
from typing import List, Optional, Tuple


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sklearn as sk
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# injected by test
dump_figure()

# EXECUTED IN 0.0023s


# ------------------------------- CODE CELL nb-3 -------------------------------
X_clusters = np.loadtxt("clusters.txt")
plt.scatter(X_clusters[:,0],X_clusters[:,1], s=0.5)
plt.show()

# injected by test
dump_figure()

# EXECUTED IN 0.353s
# STDOUT
#     save current figure: figures/fig_nb_3_1


# ------------------------------- CODE CELL nb-4 -------------------------------
def k_means(X: np.ndarray, k : int = 2, init_points : Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param X: numerical 2D numpy array, each row is a data point
    :param k: number of clusters, ignored if init_points is not None
    :param init_points: list of row indices which indicate the data points that the clusters are initialized with
             -> default is None, indicating that random points from the input data X are initialized as cluster centers
                if specified, k is chosen as the number of cluster centers, and input for k is ignored
    :return: two numpy arrays:
                - labels: 1D numpy array with cluster labels in {1,...,k}
                - centroids: 2D numpy array with k rows which denote the cluster centers
    """
    labels = np.zeros(len(X))
    centroids = init_points
    
    # if init_points are not given or are not the same length as k # choose them randomly
    if init_points == None: 
        init_points = X[np.random.randint(X.shape[0], size=k)]
    # assign x to correct clustercenter
    centroids = init_points
    centroids_temp = np.zeros_like(centroids)
    
    while_counter = 0
    while (centroids_temp != centroids).all() and while_counter < 20: 
        while_counter += 1 
        
        #for the first loop execution
        if np.amax(centroids_temp)!=0:
            centroids = centroids_temp
            
        # compute new labels
        for idx, x in enumerate(X):
            d = [np.linalg.norm(x - center) for center in centroids]
            labels[idx] = np.argmin(d)  
        
        #compute centroids 
        for idx_k, k in enumerate(centroids):
            indices = [idx for idx, x in enumerate(X) if labels[idx] == idx_k]
            if len(indices) == 0:
                centroids_temp[idx_k] = k
                continue
            mean_k = X_sum = np.sum(X[indices], axis=0)/len(X[indices])
            centroids_temp[idx_k] = mean_k
            np.round(centroids_temp, decimals=5, out=centroids_temp)
            
    return labels, centroids

# injected by test
dump_figure()

# EXECUTED IN 0.000743s


# ------------------------------- CODE CELL nb-5 -------------------------------
fig = plt.figure()
for i in range(10):
    labels, centroids = k_means(X_clusters, k=4)
    k_zero = np.asarray([x for idx, x in enumerate(X_clusters) if labels[idx] == 0])
    k_one = np.asarray([x for idx, x in enumerate(X_clusters) if labels[idx] == 1 ])
    k_two = np.asarray([x for idx, x in enumerate(X_clusters) if labels[idx] == 2 ])
    k_three = np.asarray([x for idx, x in enumerate(X_clusters) if labels[idx] == 3 ])

    
    fig.add_subplot(5, 2, i+1)
    if len(k_zero)!=0:
        plt.scatter(k_zero[:,0],k_zero[:,1], s=0.5, c="red")
    if len(k_one)!=0:
        plt.scatter(k_one[:,0],k_one[:,1], s=0.5, c="green")      
    if len(k_two)!=0:
        plt.scatter(k_two[:,0],k_two[:,1], s=0.5, c="yellow")
    if len(k_zero)!=0:
        plt.scatter(k_three[:,0],k_three[:,1], s=0.5, c="blue")

fig.savefig("clusters1b.png")
fig.show()

# injected by test
dump_figure()

# EXECUTED IN 5.2s
# STDOUT
#     save current figure: figures/fig_nb_5_1


# ------------------------------- CODE CELL nb-6 -------------------------------
# Helper function for calculating the probability in step 2
# computes the distribution of probabilities of each individual point in the data to be chosen as initial cluster centroid

def calc_probability(X: np.ndarray, curr_centroids: List[int]) -> np.ndarray:
    """
    :param X: 2D numpy array, consisting of all the data points we want to cluster
    :param curr_centroids: list of row indices which indicate the points that are already chosen as cluster centers
    :return: 1D numpy array, containing the probabilities of each point in the data to be chosen as next cluster center
    """
    
    #compute p(x)  
    d_array = []
    for idx, x in enumerate(X):
        temp = [np.linalg.norm(x - c) for c in curr_centroids]
        D = min(temp)
        d_array.append(D**2)

    p = [d/sum(d_array) for d in d_array]
        
    return p

# calc_probability(X_clusters, [12 ,  13])

# injected by test
dump_figure()

# EXECUTED IN 0.000471s


# ------------------------------- CODE CELL nb-7 -------------------------------
def init_k_means_pp(X: np.ndarray, k: int) -> List[int]:
    """
    :param X: numerical 2D numpy array, where each row represents a data point
    :param k: number of clusters 
    :
    :return: list of k row indices which indicate the data points that the clusters are initialized with
    """
    centers = []
    for i in range(k):
        if i == 0: 
            centers.append(X[np.random.randint(0,X.shape[0])].tolist())
            continue
        p = calc_probability(X,centers)
        # append new center 
        centers.append(X[np.argmax(p)].tolist())
    return centers
# init_k_means_pp(X_clusters, 5)

# injected by test
dump_figure()

# EXECUTED IN 0.000384s


# ------------------------------- CODE CELL nb-8 -------------------------------
fig = plt.figure()
for i in range(10):
    labels, centroids = k_means(X_clusters, k=4, init_points=init_k_means_pp(X_clusters, 4))
    k_zero = np.asarray([x for idx, x in enumerate(X_clusters) if labels[idx] == 0])
    k_one = np.asarray([x for idx, x in enumerate(X_clusters) if labels[idx] == 1 ])
    k_two = np.asarray([x for idx, x in enumerate(X_clusters) if labels[idx] == 2 ])
    k_three = np.asarray([x for idx, x in enumerate(X_clusters) if labels[idx] == 3 ])

    fig.add_subplot(5, 2, i+1)
    if len(k_zero)!=0:
        plt.scatter(k_zero[:,0],k_zero[:,1], s=0.5, c="red")
    if len(k_one)!=0:
        plt.scatter(k_one[:,0],k_one[:,1], s=0.5, c="green")      
    if len(k_two)!=0:
        plt.scatter(k_two[:,0],k_two[:,1], s=0.5, c="yellow")
    if len(k_zero)!=0:
        plt.scatter(k_three[:,0],k_three[:,1], s=0.5, c="blue")
        #pdf.savefig()
fig.savefig('clusters1d.png')       
fig.show()

# injected by test
dump_figure()

# EXECUTED IN 8.73s
# STDOUT
#     save current figure: figures/fig_nb_8_1


# ------------------------------- CODE CELL nb-9 -------------------------------
X_mas = np.loadtxt("xmas.txt")
plt.scatter(X_mas[:,0],X_mas[:,1], s=0.2)
plt.show()

# injected by test
dump_figure()

# EXECUTED IN 0.376s
# STDOUT
#     save current figure: figures/fig_nb_9_1


# ------------------------------ CODE CELL nb-10 -------------------------------
def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    :param X: numerical 2D numpy array, where each row represents a data point
    :param labels: 1D numpy array containing the corresponding cluster labels in {1,...,k}
    :
    :return: resulting silhouette score as float
    """
    # compute distance between points in the same class
    in_class_dist = np.zeros(X.shape[0])
    
    X_copy = X
    for idx, x in enumerate(X):
        for idxx, y in enumerate(X):
            if labels[idxx] == labels[idx]:
                in_class_dist[idx] += np.linalg.norm(x-y)
    # compute a for every point 
    a = [i/(np.count_nonzero(labels == labels[idx]) - 1) for idx, i in enumerate(in_class_dist)]
    
    # compute b for every point 
    b = np.zeros(X.shape[0])
    
    for idx, x in enumerate(X):
        for idxy, y in enumerate(X):
            if labels[idxy] != labels[idx]:
                i = 0
                temp = 0
                for idxz, z in enumerate(X):
                    if labels[idxy] == labels[idxz]:
                        temp += np.linalg.norm(z-y)
                        i += 1 
                #reasisation of the min part of the b function
                if b[idx] == 0:
                    b[idx] = temp/i
                elif b[idx] > temp/i:
                    b[idx] = temp/i; 
    
    # comput s for all points 
    s = [(b[idx] - a[idx]) / max(a[idx], b[idx])for idx, x in enumerate(X)]
                        
    return s

# injected by test
dump_figure()

# EXECUTED IN 0.000555s


# ------------------------------ CODE CELL nb-11 -------------------------------
xmas_cluster_labels = 15

# injected by test
dump_figure()

# EXECUTED IN 0.000296s


# ------------------------------ CODE CELL nb-12 -------------------------------
s_bewertung = []
s_impl_arr = []

for k in range(10,21):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_mas)
    s = sk.metrics.silhouette_score(X_mas, kmeans.labels_, metric='euclidean')
    s_bewertung.append(s)
xmas_cluster_labels = 10 + np.argmax(s_bewertung) 
print("xmas_cluster_labels = ", xmas_cluster_labels)

# injected by test
dump_figure()

# EXECUTED IN 6.39s
# STDOUT
#     xmas_cluster_labels =  15


# ------------------------------ CODE CELL nb-13 -------------------------------
## TEST
#for k in range(10,21):
#    print("k =", k)
#    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_mas)
#    s = silhouette_score(X_mas, kmeans.labels_)
#    s_bewertung.append(s)
#xmas_cluster_labels_ = 10 + np.argmax(s_bewertung)
#print("xmas_cluster_labels_ = ", xmas_cluster_labels_)

# injected by test
dump_figure()

# EXECUTED IN 0.000423s


# ------------------------------ CODE CELL nb-14 -------------------------------
df3 = pd.read_csv ("german_credit.csv")
df3.credit = df3.credit.replace(2,0)
df3 = pd.get_dummies(df3, drop_first = True)
df3.head()

# injected by test
dump_figure()

# EXECUTED IN 0.0424s


# ------------------------------ CODE CELL nb-15 -------------------------------
# absolute amount of old and young
n_old = sum(df3.age > 25) # unprotected group
n_young = sum(df3.age <= 25) # protected group
print(n_old)
print(n_young)

n_good_old = sum((df3.age > 25) & (df3.credit == 1))
n_good_young = sum((df3.age <= 25) & (df3.credit == 1))

# relative amount of old and young with good credit score
r_good_old = n_good_old / n_old
r_good_young = n_good_young / n_young
print(r_good_old)
print(r_good_young)

# disparity?
#print(r_good_young/r_good_old)

#r_bad_old = (sum((df3.age > 25) & (df3.credit == 0))) / n_old
#r_bad_young = (sum((df3.age <= 25) & (df3.credit == 0))) /n_young
#print(r_bad_old)
#print(1-r_good_old)

# injected by test
dump_figure()

# EXECUTED IN 0.00552s
# STDOUT
#     810
#     190
#     0.7283950617283951
#     0.5789473684210527


# ------------------------------ CODE CELL nb-16 -------------------------------
def relative_chance(y_pred: np.ndarray, s_arr: np.ndarray) -> float:
    """
    :param y_pred: 1D numpy array of binary predicted classes
    :param s_arr: binary 1D numpy array representing the sensitive attribute
    :
    :return: resulting relative chance score
    """

    positive_attr = s_arr==1 # if sensitive attribute: unprotected, old elif class attribute: good
    negative_attr = s_arr==0 # if sensitive attribute: protected, young elif class attribute: bad

    # 1-pr(a)=pr(a^c)
    return (np.sum(y_pred[negative_attr])/np.sum(negative_attr)) / (np.sum(y_pred[positive_attr]/np.sum(positive_attr)))


def groupwise_accuracy(y: np.ndarray, y_pred: np.ndarray, s_arr: np.ndarray) -> Tuple[float, float]:
    """
    :param y: 1D numpy array of binary true classes
    :param y_pred: 1D numpy array of binary predicted classes
    :param s_arr: binary 1D numpy array representing the sensitive attribute
    :
    :return: tuple of accuracy scores:
    :       - First element is accuracy on the instances from the protected class, 
    :       - Second element is accuracy on the instances from the unprotected class,
    """
    # TODO insert your code here

    # accuracy = (true pos. + true neg.)/all
    # ternary operator
    unprotected = s_arr==1 # positive sensitive attribute i.e. old, unprotected group
    protected = s_arr==0 # negative sensitive attribute i.e. young, protected group
    true = y==y_pred
    acc_protected = np.sum(true[protected])/np.sum(protected)
    acc_unprotected = np.sum(true[unprotected])/np.sum(unprotected)

    return acc_protected,acc_unprotected

# injected by test
dump_figure()

# EXECUTED IN 0.000474s


# ------------------------------ CODE CELL nb-17 -------------------------------
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

y = df3.credit.to_numpy()
# do not scale before test train split (avoid data leak)
X = preprocessing.scale(df3.drop("credit", axis=1) )

X_train = X[:700]
y_train = y[:700]
X_test = X[700:]
y_test = y[700:]
logit3c = LogisticRegression(max_iter = 10000).fit(X_train,y_train)
y_pred = logit3c.predict(X_test)

# injected by test
dump_figure()

# EXECUTED IN 0.0305s


# ------------------------------ CODE CELL nb-18 -------------------------------
s_arr = (df3[700:].age>25).to_numpy().astype(int)
acc_3c = np.sum(y[700:]==y_pred)/len(y_pred)
#acc_3c_test = accuracy_score(df3[700:].credit, y_pred)
gr_acc_3c = groupwise_accuracy(y[700:],y_pred,s_arr)
rc_3c = relative_chance(y_pred, s_arr)

print(acc_3c)
#print(acc_3c_test)
print(gr_acc_3c)
print(rc_3c)

# injected by test
dump_figure()

# EXECUTED IN 0.00139s
# STDOUT
#     0.78
#     (0.6724137931034483, 0.8057851239669421)
#     0.7323917828319882


# ------------------------------ CODE CELL nb-19 -------------------------------
X_new = preprocessing.scale(df3.drop(["credit" , "age"], axis=1))
X_new_train = X_new[:700]
X_new_test = X_new[700:]
logit3d = LogisticRegression(max_iter = 10000).fit(X_new_train,y_train)
y_pred = logit3d.predict(X_new_test)

# injected by test
dump_figure()

# EXECUTED IN 0.0309s


# ------------------------------ CODE CELL nb-20 -------------------------------
acc_3d = np.sum(y[700:]==y_pred)/len(y_pred)
gr_acc_3d = groupwise_accuracy(y[700:],y_pred,s_arr)
rc_3d = relative_chance(y_pred, s_arr)

print(acc_3d)
print(gr_acc_3d)
print(rc_3d)

# injected by test
dump_figure()

# EXECUTED IN 0.000558s
# STDOUT
#     0.7866666666666666
#     (0.6724137931034483, 0.8140495867768595)
#     0.7767791636096845


# ------------------------------ CODE CELL nb-21 -------------------------------
def create_massaged_labels(X: np.ndarray, s_arr: np.ndarray, y: np.ndarray, m: int, clf=LogisticRegression()) -> np.ndarray:
    """
    :param X: 2D numpy array of training features
    :param s_arr: binary 1D numpy array representing the sensitive attribute
    :param y:  binary 1D numpy array of true classes
    :param m: number of labels to flip
    :param clf: sklearn classifier that should be used to produce the confidence scores. 
    :           You may assume that it contains a fit() and predict_proba() function. 
    :           Defaults to logistic regression classifier with standard parameters
    :
    :return: 1D numpy array of massaged training labels, needs to have the same length as y
    """
    X = preprocessing.scale(df3.drop(["credit", "age"], axis=1) )
    
    logit3c = clf(max_iter = 10000).fit(X,y)
    y_score = clf.predict_proba(X)[:,1] # probability that an individual instance is classed as positive.
    df3.insert(0,"y_score", y_score, True)
    
    m_0 = df3[(s_arr <= 25) & (df3.credit == 0)].sort_values('y_score', ascending = False)
    m_1 = df3[(s_arr > 25) & (df3.credit == 1)].sort_values('y_score', ascending = True)
    
    
    for i, row in m_0.iloc[:m].iterrows():
        row['credit'] == 1 - row.credit
        
    for i, row in m_1.iloc[:m].iterrows():
        row['credit'] == 1 - row.credit
        
    return

# injected by test
dump_figure()

# EXECUTED IN 0.00048s


# ------------------------------ CODE CELL nb-22 -------------------------------
y_msg = ...

# injected by test
dump_figure()

# EXECUTED IN 0.000303s


# ------------------------------ CODE CELL nb-23 -------------------------------
logit3f = ...

# injected by test
dump_figure()

# EXECUTED IN 0.000301s


# ------------------------------ CODE CELL nb-24 -------------------------------
acc_3f = ...
gr_acc_3f = ...
rc_3f = ...

# injected by test
dump_figure()

# EXECUTED IN 0.00031s


# ------------------------------ CODE CELL nb-25 -------------------------------
df4 = pd.read_csv("ricci.csv")
df4.head()

# injected by test
dump_figure()

# EXECUTED IN 0.00558s


# ------------------------------ CODE CELL nb-26 -------------------------------
def cv_score(y_pred, s_arr):
    """
    :param y_pred: 1D numpy array of binary predicted classes
    :param s_arr: binary 1D numpy array representing the sensitive attribute
    :
    :return: resulting cv score
    """
    pos_mask = s_arr==1
    neg_mask = s_arr==0

    return 1.0 - np.sum(y_pred[pos_mask])/np.sum(pos_mask) + np.sum(y_pred[neg_mask])/np.sum(neg_mask)

# injected by test
dump_figure()

# EXECUTED IN 0.000353s


# ------------------------------ CODE CELL nb-27 -------------------------------
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X = df4.drop(labels=['Class'], axis=1)
y = df4['Class']

gaussianNB = GaussianNB()
gaussianNB_model = gaussianNB.fit(X, y)
y_pred = gaussianNB_model.predict(X)

acc_4a = accuracy_score(df4['Class'], y_pred)
print(acc_4a)

cv_4a = cv_score(y_pred, df4['Race'])
print(cv_4a)

# injected by test
dump_figure()

# EXECUTED IN 0.0137s
# STDOUT
#     0.9152542372881356
#     0.5341176470588235


# ------------------------------ CODE CELL nb-28 -------------------------------
import numpy as np
from sklearn.naive_bayes import GaussianNB

class TwoNaiveBayes:
    """
    :class implementing the Two Naive Bayes Algorithm by Calders and Verwer
    :
    :class attributes:
    :attr nb0: Gaussian Naive Bayes classifier used to make predictions on minority class
    :attr nb1: Gaussian Naive Bayes classifier used to make predictions on privileged class
    :attr delta: adaptation rate used in the iteration in step 2
    :attr eps: fairness tolerance, we consider a classification fair if the cv score is at least 1 - eps
    :
    :class methods:
    :function __init__: constructor of the class which initializes the model
    :function fit: function that fits model parameters on training data
    :function predict: function that utilizes trained model parameters to make predictions on unseen target data
    """
    
    def __init__(self, delta: Optional[float] = 0.01, eps: Optional[float] = 0.05):
        """
        : Constructor of the class, we only initialize internal parameters here.
        : Do not change anything about the constructor!
        """
        self.nb0 = GaussianNB()
        self.nb1 = GaussianNB()
        self.delta = delta
        self.eps = eps

    def fit(self, X: np.ndarray, s_arr: np.ndarray, y: np.ndarray):
        """
        :param X: 2D numerical numpy array reprenting the feature matrix without the sensitive attribute
        :param s_arr: binary 1D numpy array representing the sensitive attribute
        :param  y: binary 1D numpy array representing the class labels to train on
        :
        :return nothing, this function only fits internal parameters
        """
        
        # From funcrion 'cv_score'
        pos_mask = s_arr==1
        neg_mask = s_arr==0
        
        self.nb0.fit(X[neg_mask], y[neg_mask])
        self.nb1.fit(X[pos_mask], y[pos_mask])
    
    def predict(self, X: np.ndarray, s_arr: np.ndarray) -> np.ndarray:
        """
        :param X: 2D numerical numpy array reprenting the feature matrix without the sensitive attribute
        :param s_arr: binary 1D numpy array representing the sensitive attribute to consider when predicting
        :
        :return: 1D numpy vector of (fair) class labels
        """
        
        # From funcrion 'cv_score'
        pos_mask = s_arr==1
        neg_mask = s_arr==0
        
        pred_0 = self.nb0.predict(X[neg_mask])
        pred_1 = self.nb1.predict(X[pos_mask])
        
        pred = np.ndarray(shape=(s_arr.size,), dtype='int')
        
        pred[neg_mask] = pred_0
        pred[pos_mask] = pred_1
        
        cv = cv_score(pred, s_arr)
        
        # Skip the initilization for the while loop and the while loop, if it would not be used.
        if(cv >= (1 - self.eps)):
            return pred
        
        unique, frequency = np.unique(y, return_counts=True)
        y_frequency = dict(zip(unique, frequency))
        # Dict access to ensure correct valute for positive frequency
        n_y_1 = y_frequency[1]
        
        unique, frequency = np.unique(s_arr, return_counts = True)
        s_frequency = dict(zip(unique, frequency))
        # Dict access to ensure correct valute for positive and negative frequency
        n_s_0 = s_frequency[0]
        n_s_1 = s_frequency[1]
        
        unique, frequency = np.unique(np.column_stack((y, s_arr)), axis=0, return_counts = True)
        n_0_0 = frequency[unique.tolist().index([0, 0])]
        n_0_1 = frequency[unique.tolist().index([0, 1])]
        n_1_0 = frequency[unique.tolist().index([1, 0])]
        n_1_1 = frequency[unique.tolist().index([1, 1])]        
        
        while cv < (1 - self.eps):
            # Backup of values
            b_0_0 = n_0_0
            b_0_1 = n_0_1
            b_1_0 = n_1_0
            b_1_1 = n_1_1
            
            n_pos = len(np.where(pred == 1)[0])
            
            if(n_pos < n_y_1):
                n_1_0 = n_1_0 + self.delta * n_0_1
                n_0_0 = n_0_0 - self.delta * n_0_1
            else:
                n_0_1 = n_0_1 + self.delta * n_1_0
                n_1_1 = n_1_1 - self.delta * n_1_0
                
            if(n_0_0 < 0 or n_0_1 < 0 or n_1_0 < 0 or n_1_1 < 0):
                # Load old values
                # Only present because the algrotithm demands it.
                # Could be removed with no effect.
                n_0_0 = b_0_0
                n_0_1 = b_0_1
                n_1_0 = b_1_0
                n_1_1 = b_1_1
                # Negativity, break out of loop.
                break
                
            # Recalculate priors  
            p_0_0 = n_0_0 / n_s_0  
            p_0_1 = n_0_1 / n_s_1
            p_1_0 = n_1_0 / n_s_0
            p_1_1 = n_1_1 / n_s_1

            self.nb0.class_prior_ = [p_0_0, p_1_0]
            self.nb1.class_prior_ = [p_0_1, p_1_1]

            pred_0 = self.nb0.predict(X[neg_mask])
            pred_1 = self.nb1.predict(X[pos_mask])

            pred[neg_mask] = pred_0
            pred[pos_mask] = pred_1

            # Update cv
            cv = cv_score(pred, s_arr)
            
        
        return pred

# injected by test
dump_figure()

# EXECUTED IN 0.000764s


# ------------------------------ CODE CELL nb-29 -------------------------------
X = df4.drop(labels=['Class', 'Race'], axis=1)
y = df4['Class']
s_arr = df4['Race'].to_numpy()

twoNaiveBayes = TwoNaiveBayes()
twoNaiveBayes.fit(X, s_arr, y)
y_pred = twoNaiveBayes.predict(X, s_arr)

acc_4c = accuracy_score(df4['Class'], y_pred)
print(acc_4c)

cv_4c = cv_score(y_pred, s_arr)
print(cv_4c)

# injected by test
dump_figure()

# EXECUTED IN 1.43s
# STDOUT
#     0.8728813559322034
#     0.9547058823529411


# ------------------------------ CODE CELL nb-30 -------------------------------
# injected by test
dump_figure()

# EXECUTED IN 0.000339s


# ------------------------------ CODE CELL nb-31 -------------------------------
# injected by test
dump_figure()

# EXECUTED IN 0.000303s


# ------------------------------ CODE CELL nb-32 -------------------------------
# injected by test
dump_figure()

# EXECUTED IN 0.000304s


# ------------------------------ CODE CELL nb-33 -------------------------------
# injected by test
dump_figure()

# EXECUTED IN 0.000367s


# ------------------------ CODE CELL injected: teardown ------------------------
__IA_FLAG__ = True

# EXECUTED IN 0.000152s


