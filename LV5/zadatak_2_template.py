import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("LV5\penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'] = df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}).astype(int)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

cl_train, count_train = np.unique(y_train, return_counts=True)
cl_test, count_test = np.unique(y_test, return_counts=True)

x_bar = np.arange(len(cl_train))
width = 0.35

plt.bar(x_bar - width/2, count_train, width, label="Train")
plt.bar(x_bar + width/2, count_test, width, label="Test")
plt.xticks(x_bar, [labels[c] for c in range(len(cl_train))])
plt.legend()
plt.show()

LogReg_model = LogisticRegression()
LogReg_model.fit(X_train, y_train)
intercept = LogReg_model.intercept_
coefs = LogReg_model.coef_
print(f'Intercept: {intercept}')
print(f'Coefs: {coefs}')

plot_decision_regions(X_train, y_train.ravel(), LogReg_model)
plt.show()

y_test_p = LogReg_model.predict(X_test)

cm = confusion_matrix(y_true=y_test, y_pred=y_test_p)
print("Matrica zabune: \n", cm)

print(f'Tocnost: {accuracy_score(y_test, y_test_p):.2f}\n')

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

print(classification_report(y_test, y_test_p))

df = pd.read_csv("penguins.csv")
#input_variables = ['island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex']
input_variables = ['island']
df['species'] = df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}).astype(int)
df['island'] = df['island'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}).astype('Int64')
df['sex'] = df['sex'].map({'male': 0, 'female': 1}).astype('Int64')
df = df.dropna()


X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

LogReg_model.fit(X_train, y_train)
y_test_p = LogReg_model.predict(X_test)

print(classification_report(y_test, y_test_p))

'''
Input variables: island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        28
           1       1.00      1.00      1.00        11
           2       1.00      1.00      1.00        28

    accuracy                           1.00        67
   macro avg       1.00      1.00      1.00        67
weighted avg       1.00      1.00      1.00        67

Input variables: island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
              precision    recall  f1-score   support

           0       0.97      1.00      0.98        28
           1       1.00      0.91      0.95        11
           2       1.00      1.00      1.00        28

    accuracy                           0.99        67
   macro avg       0.99      0.97      0.98        67
weighted avg       0.99      0.99      0.98        67

Input variables: island, bill_length_mm, bill_depth_mm, flipper_length_mm
              precision    recall  f1-score   support

           0       0.97      1.00      0.98        28
           1       1.00      0.91      0.95        11
           2       1.00      1.00      1.00        28

    accuracy                           0.99        67
   macro avg       0.99      0.97      0.98        67
weighted avg       0.99      0.99      0.98        67

Input variables: island, bill_length_mm, bill_depth_mm
              precision    recall  f1-score   support

           0       0.97      1.00      0.98        28
           1       0.90      0.82      0.86        11
           2       0.96      0.96      0.96        28

    accuracy                           0.96        67
   macro avg       0.94      0.93      0.93        67
weighted avg       0.95      0.96      0.95        67

Input variables: island, bill_length_mm
              precision    recall  f1-score   support

           0       0.93      0.96      0.95        28
           1       1.00      0.91      0.95        11
           2       0.96      0.96      0.96        28

    accuracy                           0.96        67
   macro avg       0.97      0.95      0.95        67
weighted avg       0.96      0.96      0.96        67

Input variables: island
              precision    recall  f1-score   support

           0       0.38      0.61      0.47        28
           1       0.50      1.00      0.67        11
           2       0.00      0.00      0.00        28

    accuracy                           0.42        67
   macro avg       0.29      0.54      0.38        67
weighted avg       0.24      0.42      0.30        67
'''