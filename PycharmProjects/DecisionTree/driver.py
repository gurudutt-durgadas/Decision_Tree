from DecisionTree import *
import pandas as pd
from sklearn import model_selection


#header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
#df = pd.read_csv('ftp://ftp.ics.uci.edu/pub/machine-learning-databases/iris/iris.data')
#df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/d546eaee765268bf2f487608c537c05e22e4b221/iris.csv')
df = pd.read_csv('http://bit.ly/IrisDataSet')

header = list(df.columns.values)
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
accu = computeAccuracy(test, t)
print("Accuracy on test = " + str(accu))

## TODO: You have to decide on a pruning strategy

plistItems = pruneList(leaves)
print(plistItems)
for item in plistItems:
    t_pruned = prune_tree(t, [item])

    acc = computeAccuracy(test, t_pruned)
    if(accu<acc):
        print("*************Tree after pruning*******")
        print_tree(t_pruned)
        print("Accuracy on test = " + str(acc))
        break
    print("Accuracy on test = " + str(acc))
    t_pruned = build_tree(train, header)


