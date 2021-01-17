# Dataset from:
# https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# Neural network (MLPClassifier)
def neural():
    mlp = MLPClassifier(hidden_layer_sizes=(11, 11, 11), max_iter=1000)
    mlp.fit(trainwinedata, traintested)
    score = mlp.score(testwinedata, testtested)
    print(score)
    matrix(mlp, "sieć")
    return score, "Neural"


# Decision tree (DecisionTreeClassifier)
def dectree():
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(trainwinedata, traintested)
    score = decision_tree.score(testwinedata, testtested)
    print(score)
    matrix(decision_tree, "drzewo")
    return score, "Drzewo decyzyjne"


# K nearest neighbours (KNeighborsClassifier)
def knn(n):
    res = neighbors.KNeighborsClassifier(n, weights='distance')
    res.fit(trainwinedata, traintested)
    score = res.score(testwinedata, testtested)
    print(score)
    s = "k-NN, k = " + str(n)
    matrix(res, s)
    return score, s


# Naive Bayes (GaussianNB)
def naive_bayes():
    nb = GaussianNB()
    nb.fit(trainwinedata, traintested)
    score = nb.score(testwinedata, testtested)
    print(score)
    matrix(nb, "naive bayes")
    return score, "Naiwny bayesowski"


# Plotting confusion matrix
def matrix(classf, text):
    m = plot_confusion_matrix(classf, testwinedata, testtested, normalize=None)
    s = "Macierz błędu: " + text
    m.ax_.set_title(s)
    plt.show()


# Finding a classifier with best accuracy
def accuracies():
    check = 0
    name = ""
    for i in range(0, len(sclist)):
        if sclist[i] > check:
            check = sclist[i]
            name = stlist[i]
    print("Klasyfikator ", name, ", skuteczność: ", check)


# Opening csv
df = pd.read_csv('winequality-red.csv')
print(df.isnull().sum())
winedata = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
               'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values
tested = df['quality'].values

# Splitting the data into test and train sets
trainwinedata, testwinedata, traintested, testtested = train_test_split(winedata, tested, test_size=0.33, random_state=1)

# Running the functions and gathering the data for the bar graph
sclist = []
stlist = []
sc, st = knn(3)
sclist.append(sc)
stlist.append(st)
sc, st = knn(5)
sclist.append(sc)
stlist.append(st)
sc, st = knn(11)
sclist.append(sc)
stlist.append(st)
sc, st = naive_bayes()
sclist.append(sc)
stlist.append(st)
sc, st = dectree()
sclist.append(sc)
stlist.append(st)
sc, st = neural()
sclist.append(sc)
stlist.append(st)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(stlist, sclist)
ax.set_ylabel('Skuteczność')
ax.set_xlabel('Klasyfikatory')
accuracies()
for i in range(0, len(stlist)):
    plt.annotate(str(stlist[i]), xy=(stlist[i], sclist[i]), ha='center', va='bottom')
plt.show()
