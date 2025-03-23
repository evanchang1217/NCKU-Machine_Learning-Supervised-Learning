#less flexible larger bias -> less variance
import math
import pickle
import gzip
import numpy as np
import matplotlib.pylab as plt
#%matplotlib inline

# importing all the required libraries

from math import exp
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

class MNIST_import:
    """
    sets up MNIST dataset from OpenML 
    """
    def __init__(self):
        
        df = pd.read_csv("mnist_784.csv")
        
        # Create arrays for the features and the response variable
        # store for use later 
        y = df['class'].values
        X = df.drop('class', axis=1).values #class?????????????????????????????????????????????????????????
        
        # Convert the labels to numeric labels
        y = np.array(pd.to_numeric(y))
        
        # create training and validation sets 
        self.train_x, self.train_y = X[:1500,:], y[:1500]
        self.val_x, self.val_y = X[1500:2000,:], y[1500:2000]
        
data = MNIST_import()

def view_digit(x, label=None):
    fig = plt.figure(figsize=(3,3))
    plt.imshow(x.reshape(28,28), cmap='gray');
    plt.xticks([]); plt.yticks([]);
    if label: plt.xlabel("true: {}".format(label), fontsize=16)

#1----------------------------------------------------- part1
#Display a particular digit using the above function:
training_index = 9

# your code here
# Here are the numbers you need to provide here:
num_training_examples = 0
num_test_examples = 0
pixels_per_image = 0

# your code here

# 計算每個圖像中的像素數量
pixels_per_image = data.train_x.shape[1]

# 計算訓練集中的示例數量
num_training_examples = data.train_x.shape[0]

# 計算測試集中的示例數量
num_test_examples = data.val_x.shape[0]

print(num_training_examples)
print(num_test_examples)
print(pixels_per_image)

#1------------------------------------------------------part2
class KNN:
    """
    Class to store data for regression problems 
    """
    def __init__(self, x_train, y_train, K=5):
        """
        Creates a kNN instance

        :param x_train: numpy array with shape (n_rows,1)- e.g. [[1,2],[3,4]]
        :param y_train: numpy array with shape (n_rows,)- e.g. [1,-1]
        :param K: The number of nearest points to consider in classification
        """
        
        # Import and build the BallTree on training features 
        from sklearn.neighbors import BallTree
        self.balltree = BallTree(x_train)
        
        # Cache training labels and parameter K 
        self.y_train = y_train
        self.K = K 
        
        
    def majority(self, neighbor_indices, neighbor_distances=None):
        """
        Given indices of nearest neighbors in training set, return the majority label. 
        Break ties by considering 1 fewer neighbor until a clear winner is found. 

        :param neighbor_indices: The indices of the K nearest neighbors in self.X_train 
        :param neighbor_distances: Corresponding distances from query point to K nearest neighbors. 
        """
        
        # your code here
        k = self.K
        while k > 0:
            # Get labels of nearest neighbors
            labels = self.y_train[neighbor_indices[:, :k]] #labels 变量存储了查询点 x 的 K 个最近邻居的标签
            # Count occurrences of each label
            unique_labels, counts = np.unique(labels, return_counts=True)
            # Find the label with the highest count
            max_count_label = unique_labels[np.argmax(counts)]
            # Check if there's a clear winner
            if np.sum(counts == np.max(counts)) == 1:
                return max_count_label
            else:
                k -= 1
        # If there's still a tie after reducing k to 1, return the first label
        return max_count_label
            
        
    def classify(self, x):
        """
        Given a query point, return the predicted label 
        
        :param x: a query point stored as an ndarray  
        """
        # your code here
        dist, ind = self.balltree.query([x], k=self.K)
        #ind 是 (1, K) 的 二維陣列
        return self.majority(ind)
        
    def predict(self, X):
        """
        Given an ndarray of query points, return yhat, an ndarray of predictions 

        :param X: an (m x p) dimension ndarray of points to predict labels for 
        """
        # your code here
        y_pred = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            y_pred[i] = self.classify(X[i])
        return y_pred
    
#-------------------------------------------------------------------------peer review


# 使用 K=3 进行 KNN 分类
knn = KNN(data.train_x, data.train_y, K=3)
val_yhat = knn.predict(data.val_x)

# 创建混淆矩阵
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(data.val_y, val_yhat)

print('-----conf_matrix----')
print(conf_matrix)


#peer------------------------------------------------------------------------------------------------------
'''
acc = []
wacc = []
allks = range(1, 30)

for k in allks:
    knn = KNN(data.train_x, data.train_y, K=k)
    val_yhat = knn.predict(data.val_x)
    accuracy = np.mean(val_yhat == data.val_y)
    acc.append(accuracy)

# 创建准确率图表
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))
ax.plot(allks, acc, marker="o", color="steelblue", lw=3, label="unweighted")
ax.set_xlabel("number neighbors", fontsize=16)
ax.set_ylabel("accuracy", fontsize=16)
plt.xticks(range(1, 31, 2))
ax.grid(alpha=0.25)
plt.savefig('week4-2.png')
plt.show()
'''

#2-parta-----------------------------------------------------------------------------------------------------------

def get_spam_dataset(filepath="spamdata.csv", test_split=0.1):
    '''
    get_spam_dataset
    
    Loads csv file located at "filepath". Shuffles the data and splits
    it so that the you have (1-test_split)*100% training examples and 
    (test_split)*100% testing examples.
    
    Args:
        filepath: location of the csv file
        test_split: percentage/100 of the data should be the testing split
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names
        Note: feature_names is a list of all column names including isSpam.
        
        (in that order)
        first four are  np.ndarray
        
    '''
    
    # your code here
    # 讀取CSV文件
    data = pd.read_csv(filepath,sep = ' ')
    
    # 從數據中分離特徵和目標
    X = data.drop(columns=['isSPAM']).values
    y = data['isSPAM'].values
    
    # 使用sklearn中的train_test_split函數進行數據集拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    
    # 獲取特徵名稱
    feature_names = data.columns.tolist()
    
    return X_train, X_test, y_train, y_test, feature_names


# TO-DO: import the data set into five variables: X_train, X_test, y_train, y_test, label_names
# Uncomment and edit the line below to complete this task.

test_split = 0.1 # default test_split; change it if you'd like; ensure that this variable is used as an argument to your function
# your code here

#X_train, X_test, y_train, y_test, label_names = np.arange(5)
X_train, X_test, y_train, y_test, label_names = get_spam_dataset(test_split=test_split)
print("X_train 形状：", X_train.shape)
print("X_test 形状：", X_test.shape)
print("y_train 形状：", y_train.shape)
print("y_test 形状：", y_test.shape)

#2-partb----------------------------------------------------------------------------------------------------------------------------------
def build_dt(data_X, data_y, max_depth = None, max_leaf_nodes =None):
    '''
    This function does the following:
    1. Builds the decision tree classifier using sklearn 
    2. Fits it to the provided data.
    
    
    Arguments
        data_X - a np.ndarray
        data_y - np.ndarray
        max_depth - None if unrestricted, otherwise an integer for the maximum
                depth the tree can reach.
    
    Returns:
        A trained DecisionTreeClassifier
    '''
    
    # your code here
    # 构建决策树分类器
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    
    # 将分类器拟合到提供的数据
    dt_classifier.fit(data_X, data_y)
    
    return dt_classifier

#peer review-------------------------------------------------------------------------------------------------

def calculate_precision(y_true, y_pred, pos_label_value=1.0):
    '''
    This function accepts the labels and the predictions, then
    calculates precision for a binary classifier.
    
    Args
        y_true: np.ndarray
        y_pred: np.ndarray
        
        pos_label_value: (float) the number which represents the postiive
        label in the y_true and y_pred arrays. Other numbers will be taken
        to be the non-positive class for the binary classifier.
    
    Returns precision as a floating point number between 0.0 and 1.0
    '''
    
    
    # 计算真阳性、假阳性和精度
    true_positive = np.sum((y_true == pos_label_value) & (y_pred == pos_label_value))
    false_positive = np.sum((y_true != pos_label_value) & (y_pred == pos_label_value))
    
    # 如果没有预测为正类别的样本，则返回 0
    if true_positive + false_positive == 0:
        return 0.0
    
    # 计算并返回精度
    precision = true_positive / (true_positive + false_positive)
    return precision


def calculate_recall(y_true, y_pred, pos_label_value=1.0):
    '''
    This function accepts the labels and the predictions, then
    calculates recall for a binary classifier.
    
    Args
        y_true: np.ndarray
        y_pred: np.ndarray
        
        pos_label_value: (float) the number which represents the postiive
        label in the y_true and y_pred arrays. Other numbers will be taken
        to be the non-positive class for the binary classifier.
    
    Returns precision as a floating point number between 0.0 and 1.0
    '''
    
    # 计算真阳性、假阴性和召回率
    true_positive = np.sum((y_true == pos_label_value) & (y_pred == pos_label_value))
    false_negative = np.sum((y_true == pos_label_value) & (y_pred != pos_label_value))
    
    # 如果没有真阳性的样本，则返回 0
    if true_positive + false_negative == 0:
        return 0.0
    
    # 计算并返回召回率
    recall = true_positive / (true_positive + false_negative)
    return recall

#peer--------------------------------------------------------------------------------------------------------
#創建max_depth = 2 的決策樹
shallow_dt = build_dt(X_train, y_train, max_leaf_nodes = 400)

y_pred_test = shallow_dt.predict(X_test)

precision = calculate_precision(y_test, y_pred_test, pos_label_value=1.0)
recall = calculate_recall(y_test, y_pred_test, pos_label_value=1.0)

tree_depth = shallow_dt.get_depth()

print("Precision on test set:", precision)
print("Recall on test set:", recall)
print("Depth of the tree:", tree_depth)


#peer--------------------------------------------------------------------------------------------------------------------------
# 建立一个决策树模型 dt
dt = build_dt(X_train, y_train)

# 获取成本复杂度修剪路径
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# 创建保存不同 alpha 值的分类器的向量
clfs = []

# 迭代不同的 alpha 值
for ccp_alpha in ccp_alphas:
    # 创建决策树分类器，并设置 ccp_alpha 参数
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    # 使用训练数据拟合分类器
    clf.fit(X_train, y_train)
    # 将训练好的分类器添加到向量中
    clfs.append(clf)

# 打印最后一个决策树的节点数量和对应的 alpha 值
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

# 生成训练集和测试集的得分并绘制其随着 ccp_alpha 值变化的曲线
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs Alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig('week4-5.png')
plt.show() 

