import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap="bwr")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", marker='x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

intercept = LogRegression_model.intercept_[0]
coefs = LogRegression_model.coef_[0]
print(f'Granica odluke: {intercept} + {coefs[0]}x1 + {coefs[1]}x2 = 0')

x1 = np.linspace(X_train[:, 0].min(), X_train[:, 0].max())
x2 = -(intercept + coefs[0]*x1) / coefs[1] # iz formule granice odluke

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap="bwr")
plt.plot(x1, x2, color='green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

y_test_p = LogRegression_model.predict(X_test)

cm = confusion_matrix(y_true=y_test, y_pred=y_test_p)
print("Matrica zabune: \n", cm)

tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]

print(f'Tocnost: {accuracy_score(y_test, y_test_p):.2f}\n'+
      f'Preciznost: {tp/(tp + fp):.2f}\n'+ # udio tocno klas primjera kao poz
      f'Odziv: {tp/(tp+fn):.2f}\n') # udio tocno klas primjera u skupu poz

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

colors = []
for i in range(len(y_test)):
    if y_test[i] != y_test_p[i]:  # False
        colors.append('black')
    elif y_test[i] == 0:
        colors.append('blue')   # klasa 0
    else:
        colors.append('red') # klasa 1

plt.scatter(X_test[:, 0], X_test[:, 1], c=colors)
plt.plot(x1, x2, c='purple')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()