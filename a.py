from math import exp
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#A
def gen_logistic(x, w=1, b=0):
    """
    outputing the logistic output for an input x
    :param x: scalar or numpy array of shape (n_samples, n_features). If only one feature, it must have the shape of (n_samples,1).
    :param w: weight(s); either scalar or numpy array of shape (1, n_features)
    :param b: bias; either scalar or numpy array of shape (1,)
    returns y of shape (n_samples,)
    """    
    # TODO: Finish this function to return the output of applying the sigmoid
    # function to the input x (Please do not use external libraries) store 
    # the output in y and return y. Do not change the default parameter values.
    # Hint: This function will be used in any input shape scalar (0d), 1d vector, and 2d arrays. Please make sure it can handle all those. Following reshaping codes might help.
    # Hint2: You may use design matrix using concatenation, but it is not necesary.
    
    y =0 
    if np.isscalar(x):
      x = np.array(x).reshape((1,1))
    if np.isscalar(w):
      w = np.array(w).reshape((1,1))
    if np.isscalar(b):
      b = np.array(b).reshape((1,1))  
    if b.shape==(1,):
      b= b.reshape((1,1))  

    # your code here
      
    z = np.dot(x, w.T) + b
    # 计算逻辑函数的输出    也就是算出每個輸入資料X的其為1的機率   X 為一個矩陣 (num_of_data, num_of_feature_of_eachdata)  W(1,num_of_feature_of_eachdata)
    y = 1 / (1 + np.exp(-z))
    
    
    print(y.reshape(y.shape[0],))
    return y.reshape(y.shape[0],)

#B
# your code here

# TODO: change the values of N, a and b below to check how the output of your function works
# Use a value for N greater than 1 and any limits a and b so that an S-shape graph is generated


#peer
N = 1000
Xa = -5
Xb = 5
w = 1
b = 0

x = np.expand_dims(np.linspace(Xa, Xb, N), axis=1)
y = gen_logistic(x, w, b)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))
ax.plot(x, y, lw=2)
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("y", fontsize=16)
ax.set_title("Logistic/Sigmoid Function", fontsize=16)
plt.savefig('1.png')
plt.show()

#1
sharp_transition = 'True'
#2
x_decreases_by_1 = 'True'

#C
# Importing the breast-cancer dataset from sklearn datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class BC_data:
    """
    class to import the breast cancer dataset from sklearn
    
    """
    def __init__(self):
        
        x, y = load_breast_cancer(return_X_y= True)
        self.x_train = None 
        self.x_test = None 
        self.y_train = None 
        self.y_test = None
        
        # TODO: Split the data into training and test data (use train_test_split sklearn) 
        # such that the test data size is 25% of total number of observations
        # No need to rescale the data. Use the data as is.
        # Use random_state = 5
        
        # your code here
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25, random_state=5)
        
        
data = BC_data()



from sklearn.linear_model import LogisticRegression

# 创建Logistic Regression模型并设置solver为'liblinear'
LogReg = LogisticRegression(solver='liblinear')

# 使用训练数据训练模型
LogReg.fit(data.x_train, data.y_train)




#peer
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 使用测试数据预测概率
y_pred_prob = LogReg.predict_proba(data.x_test)[:, 1]

# 计算 ROC 曲线的真正率（True Positive Rate）和假正率（False Positive Rate）
fpr, tpr, thresholds = roc_curve(data.y_test, y_pred_prob)

# 计算 AUC（Area Under the Curve）
auc = roc_auc_score(data.y_test, y_pred_prob)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('2.png')
plt.show()


#D
def calculate_precision(y_true, y_pred, pos_label_value=1.0):
    '''
    This function accepts the labels and the predictions, then
    calculates precision for a binary classifier.
    
    Args
        y_true: np.ndarray
        y_pred: np.ndarray
        
        pos_label_value: (float) the number which represents the positive
        label in the y_true and y_pred arrays. Other numbers will be taken
        to be the non-positive class for the binary classifier.
    
    Returns precision as a floating point number between 0.0 and 1.0
    '''
    true_positives = np.sum((y_true == pos_label_value) & (y_pred == pos_label_value))
    total_predicted_positives = np.sum(y_pred == pos_label_value)
    
    if total_predicted_positives == 0:
        return 0.0
    
    precision = true_positives / total_predicted_positives
    return precision

def calculate_recall(y_true, y_pred, pos_label_value=1.0):
    '''
    This function accepts the labels and the predictions, then
    calculates recall for a binary classifier.
    
    Args
        y_true: np.ndarray
        y_pred: np.ndarray
        
        pos_label_value: (float) the number which represents the positive
        label in the y_true and y_pred arrays. Other numbers will be taken
        to be the non-positive class for the binary classifier.
    
    Returns recall as a floating point number between 0.0 and 1.0
    '''
    true_positives = np.sum((y_true == pos_label_value) & (y_pred == pos_label_value))
    total_actual_positives = np.sum(y_true == pos_label_value)
    
    if total_actual_positives == 0:
        return 0.0
    
    recall = true_positives / total_actual_positives
    return recall


#peer
y_pred = gen_logistic(data.x_test, LogReg.coef_, LogReg.intercept_)

# 將概率轉換為二進制預測
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# 計算並打印精度和召回率
precision = calculate_precision(data.y_test, y_pred_binary)
recall = calculate_recall(data.y_test, y_pred_binary)

print('Model Precision : %0.2f' % precision)
print('Model Recall : %0.2f' % recall)