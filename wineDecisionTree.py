import numpy as np
from collections import Counter
import csv

class Node(object):
    def __init__(self, label, value):
        self.left = None
        self.right = None
        self.label = label
        self.data = value       

class Tree(object):
    def __init__(self, root):
        self.root = root

    def getRoot(self):
        return self.root
    
    def printTree(self, start, traversal):
        if (start):
            traversal += (str(start.label) + " ")
            traversal = self.printTree(start.left, traversal)
            traversal = self.printTree(start.right, traversal)
        return traversal

    def find(self, valid_row):
        if self.root:
            return self._find(valid_row, self.root)
        else:
            return None
        
    def _find(self, valid_row, cur_node):
        if (cur_node.data == -9999): # LEAF
            return cur_node.label
        else:
            if (valid_row[cur_node.label] >= cur_node.data and cur_node.right):
                return self._find(valid_row, cur_node.right)
            elif (valid_row[cur_node.label] < cur_node.data and cur_node.left):
                return self._find(valid_row, cur_node.left)
##########################
def gini(y, classes=[1,2,3]):
    y = np.array(y) # Convert y into numpy array
    p = 0.0 #accumulated_gini
    n = len(y)
    array = []
    for row in y:
        array = np.append(array, row[0])
    
    for label in classes:
        p += (sum(array==label)/n) ** 2 if n!= 0 else 0
    return 1 - p

def average_gini(groups, classes):
    left_gini = gini(groups[0], classes)
    right_gini = gini(groups[1], classes)
    n = len(groups[0]) + len(groups[1])
    gini_index = left_gini * len(groups[0])/n + right_gini* len(groups[1])/n
    return gini_index

def split(index, value, dataset):
    left = []
    right = []
    for row in dataset:
        if (row[index] < value):
            left.append(row)     
        else:
            right.append(row)
    left = np.array(left)
    right = np.array(right)
    return left, right

def best_split(dataset, classes):
    parent_gini = gini(dataset, classes)
    best_col = 999;
    split_point = -1;
    min_gini = 1;
    best_group = None
    row = len(dataset)
    column = len(dataset[0])
    for col in range(1,column):
        dataset = dataset[np.argsort(dataset[:,col])]
        for i in range(row-1):
            mid = (dataset[i,col] + dataset[i+1, col])/2
            groups = split(col, mid, dataset)
            avg_gini = average_gini(groups, classes)
            if (avg_gini < min_gini):
                best_col = col
                split_point = mid
                min_gini = avg_gini
                best_group = groups
    if (min_gini < parent_gini):
        parent_gini = min_gini
        return [best_col, split_point, min_gini, best_group]
    else:
        return 
   
def isSameClass(data):
    array = []
    for row in data:
        array = np.append(array, row[0])
    b = Counter(array)
    count = 0
    max_occur = 0
    label = 0
    for i in range(1,4):
        if (b[i] > 0):
            count +=1
            if (b[i] > max_occur):
                max_occur = b[i]
                label = i
    if count > 1 :
        return [False, label]
    else:
        return [True, label]

##def grow_tree(data, thisdict, classes): #--> print name of attributes
def grow_tree(data, classes):
    label = isSameClass(data)
    if(label[0]):
        leaf = Node(label[1],-9999) # leaf has no value -> set to -9999
        return leaf
    else:
        result_split = best_split(data, classes)
        if (len(result_split) == 0):
            leaf = Node(label[1], label[1])
            return leaf
        else:
##            root = Node(thisdict[str(result_split[0])], result_split[1]) #----> print name of attributes
            root = Node(result_split[0], result_split[1])
            groups = result_split[3]
            root.left = grow_tree(groups[0], classes)
            root.right = grow_tree(groups[1], classes)
    return root
#########################################################################
######################### MAIN ##########################################
#########################################################################


data = np.genfromtxt('wine_data.csv', delimiter=',')
# data dimension
i,j = data.shape

classes = [1,2,3]

thisdict={"1": "Alcohol", "2": "Malic acid", "3": "Ash", "4": "Alcalinity of ash",
          "5": "Magnesium", "6": "Total phenols", "7": "Flavanoids", "8": "Nonflavanoid phenols",
          "9": "Proanthocyanins", "10": "Color intensity", "11": "Hue",
          "12": "OD280/OD315 of diluted wines", "13": "Proline"}

#split data into training set and validation set
validation = data[0]
training = []
for i in range(1,i):
    if (i%5 == 0):
        validation = np.vstack((validation,data[i]))
    else:
        if (i == 1):
            training = data[1]
        else:
            training = np.vstack((training,data[i]))

#Grow Tree
tree = Tree(grow_tree(training, classes))
print(tree.printTree(tree.root, " "))

def predict_result (class_num):
    predict = []
    for row in class_num:
        result = tree.find(row)
        predict.append(result)
    return predict

f = open("output.txt","w")

# VALIDATION SET
report = "\tVALIDATION SET\n"
validation = validation[np.argsort(validation[:,0])]
actual = [row[0] for row in validation]
print("Actual: \n", actual)
predict = predict_result(validation)
print("Predict :\n",predict)
# Pair(predict, actual)
pair = np.vstack((predict, actual)).T
fn = [0,0,0]
fp = [0,0,0]
tp = [0,0,0]
tn = [0,0,0]
for i in range(1,4):
    for row in pair:
        if (row[0] == i and row[1] == i):
            tp[i-1] += 1
        elif (row[0] != i and row[1] != i):
            tn[i-1] += 1
        elif (row[0] == i and row[1] != row[0]):
            fp[i-1] += 1
        elif (row[0] != row[1] and row[1] == i):
            fn[i-1] += 1
    report += "\nClass %d has:\nTrue Positive: %d\nTrue Negative: %d\nFalse Positive: %d\nFalse Negative: %d\nOverall Accuracy = %f"%(i,tp[i-1],tn[i-1],fp[i-1],fn[i-1],
          (tp[i-1] + tn[i-1])/(tp[i-1]+tn[i-1]+fn[i-1]+fp[i-1]))
    report += "\nSensitivity = %f\nSpecificity = %f\n" %(tp[i-1]/(tp[i-1]+fn[i-1]),(tn[i-1]/(fp[i-1]+tn[i-1]))) 

### TRAINING SET
##report += "\n\n\tTraining Set\n"
##training = training[np.argsort(training[:,0])]
##actual = [row[0] for row in training]
##print("Actual: \n", actual)
##predict = predict_result(training)
##print("Predict :\n",predict)
### Pair(predict, actual)
##pair = np.vstack((predict, actual)).T
##fn = [0,0,0]
##fp = [0,0,0]
##tp = [0,0,0]
##tn = [0,0,0]
##for i in range(1,4):
##    for row in pair:
##        if (row[0] == i and row[1] == i):
##            tp[i-1] += 1
##        elif (row[0] != i and row[1] != i):
##            tn[i-1] += 1
##        elif (row[0] == i and row[1] != row[0]):
##            fp[i-1] += 1
##        elif (row[0] != row[1] and row[1] == i):
##            fn[i-1] += 1
##    report += "\nClass %d has:\nTrue Positive: %d\nTrue Negative: %d\nFalse Positive: %d\nFalse Negative: %d\nOverall Accuracy = %f"%(i,tp[i-1],tn[i-1],fp[i-1],fn[i-1],
##          (tp[i-1] + tn[i-1])/(tp[i-1]+tn[i-1]+fn[i-1]+fp[i-1]))
##    report += "\nSensitivity = %f\nSpecificity = %f\n" %(tp[i-1]/(tp[i-1]+fn[i-1]),(tn[i-1]/(fp[i-1]+tn[i-1]))) 

f.write(report)
f.close()
