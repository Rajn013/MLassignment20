#!/usr/bin/env python
# coding: utf-8

# 1. What is the underlying concept of Support Vector Machines?
# 

# SVM or support vector machines is a linear model for classification and regression problems.it can solve linear and non - linear problems and work well for many practical problems. The ideas of svm is simple :The alogrithum creates a line or a hyperplane which seperates the data intoo class

# 2. What is the concept of a support vector?
# 

# A support vectors machines(svm) is a types of deep learning alogrithum that performs superviserd learning for classification or regression of data groups. In AI and machine learnign, supervised learnign system provide both input and desired output data , which are labeled for classificaion

# 3. When using SVMs, why is it necessary to scale the inputs?
# 

# support vector machines (svm) optimization occurs by minimizing the desicion vector w , the optimal hyperplane is influenced by the scale of the input features and its therfore recommended that data be standardized (means0 var1) prrior to svm model tranning.

# 4. When an SVM classifier classifies a case, can it output a confidence score? What about a percentage chance?
# 

# an svm classifier can output the distance between the test instance abd the descison boundry and you can use this as a confidence score .however this score cannot be directly convert into an estimation of class probability.
# 
# Linear SVC (best performance): 92 %

# 5. Should you train a model on a training set with millions of instances and hundreds of features using the primal or dual form of the SVM problem?
# 

# if there are million of instances , you should definietly use the primal form,because the dual form will be much too slow .32 ,say you tranied as svm classifier with an rbf kernel.

# 6. Let's say you've used an RBF kernel to train an SVM classifier, but it appears to underfit the training collection. Is it better to raise or lower (gamma)? What about the letter C?
# 

# Gamma and c values are key hyperparameters that can be used to train the most optimal SVM model using RBF kernel. Higher value of gamma will means that radium of influence is limited to only support vectors. this would essentially means that the model tries and overfit.

# 7. To solve the soft margin linear SVM classifier problem with an off-the-shelf QP solver, how should the QP parameters (H, f, A, and b) be set?
# 

# #To solve the soft margin linear SVM classifier problem using an off-the-shelf Quadratic Programming (QP) solver, you need to set the QP parameters appropriately.
# #H (the Hessian matrix):the Hessian matrix H is typically a diagonal matrix where each diagonal element represents the weight assigned to a specific training instance. The weight is determined by the regularization parameter C and the kernel function (in the case of non-linear SVMs).
# 
# #for example:
# 
# import numpy as np
# n_sample = x.shape[0]
# H = np.eyes(n_sample)
# 
# #f (the linear coefficient vector): The linear coefficient vector f corresponds to the linear term in the objective function of the QP problem. For the soft margin linear SVM classifier, f is typically a vector of negative ones of length equal to the number of training instances.
# 
# #for example:
# 
# import numpy as np 
# 
# y = 2* y-1
# A = y.reshape(1, -1)
# 
# #b (the constraint vector): The constraint vector b represents the right-hand side values of the equality constraints. 
# 
# #for example:
# 
# 
# import numpy as np 
# 
# b  = np.zeros(1)

# 8. On a linearly separable dataset, train a LinearSVC. Then, using the same dataset, train an SVC and an SGDClassifier. See if you can get them to make a model that is similar to yours.
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

linear_svc = LinearSVC()
linear_svc.fit(X, y)

svc = SVC(kernel='linear')
svc.fit(X, y)

sgd = SGDClassifier(loss='hinge', alpha=0.01)
sgd.fit(X, y)

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.title('LinearSVC')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
plt.plot([-2, 2], [(-linear_svc.coef_[0][0] * -2 - linear_svc.intercept_[0]) / linear_svc.coef_[0][1],
                    (-linear_svc.coef_[0][0] * 2 - linear_svc.intercept_[0]) / linear_svc.coef_[0][1]], 'k-')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.title('SVC')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
plt.plot([-2, 2], [(-svc.coef_[0][0] * -2 - svc.intercept_[0]) / svc.coef_[0][1],
                   (-svc.coef_[0][0] * 2 - svc.intercept_[0]) / svc.coef_[0][1]], 'k-')

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.title('SGDClassifier')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
plt.plot([-2, 2], [(-sgd.coef_[0][0] * -2 - sgd.intercept_[0]) / sgd.coef_[0][1],
                   (-sgd.coef_[0][0] * 2 - sgd.intercept_[0]) / sgd.coef_[0][1]], 'k-')

plt.tight_layout()
plt.show


# 9. On the MNIST dataset, train an SVM classifier. You'll need to use one-versus-the-rest to assign all 10 digits because SVM classifiers are binary classifiers. To accelerate up the process, you might want to tune the hyperparameters using small validation sets. What level of precision can you achieve?
# 

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from keras.datasets import mnist

(X_trainval, y_trainval), (X_test, y_test) = mnist.load_data()

X_trainval = X_trainval.reshape(X_trainval.shape[0], -1).astype(np.float32) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', decision_function_shape='ovr')  # Linear kernel for linearly separable data
svm.fit(X_train, y_train)

y_pred_val = svm.predict(X_val)

precision = precision_score(y_val, y_pred_val, average='weighted')
print("Precision on validation set:", precision)

y_pred_test = svm.predict(X_test)

precision_test = precision_score(y_test, y_pred_test, average='weighted')
print("Precision on test set:", precision_test)


# 10. On the California housing dataset, train an SVM regressor.
# 

# In[1]:


from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_classifier = SGDClassifier()
sgd_classifier.fit(X_train, y_train)

y_pred = sgd_classifier.predict(X_test)

confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", confusion_mat)
print("\nClassification Report:\n", classification_rep)


# In[ ]:




