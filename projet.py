from math import sqrt 
import random
import csv
from csv import reader
from sklearn.model_selection import KFold
from random import seed
from random import randrange


def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		i = 0
		for row in csv_reader:
			if not row:
				continue
			if i != 0 :
				dataset.append(row)
			i+=1
	return dataset

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
	for row in dataset:
		row[column] = int(row[column].strip())


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)):#-1):  on exlue pas la derniere colonne parceque dans test on a pas la classe
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Split a dataset into k folds
def cross_validation_split1(dataset, n_folds):
	#random.seed(len(dataset))
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	fold = list()
	while len(fold) < fold_size:
		#nombre pseudo aleatoire<len dataset
		index = random.randrange(len(dataset_copy))
		fold.append(dataset_copy.pop(index))

	return fold,dataset_copy

def cross_validation_split(dataset, n_folds):  #
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) #* 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, n):
	folds = cross_validation_split(dataset, n_folds)  #split :ellse devise data set en n_folds list et elle renvoie une de liste de n_folds liste
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			# row_copy[-1] = None
		predicted = algorithm(train_set, test_set, n)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

def best_k (train,algorithm,n_folds):
    s=0
    k_scores = list()
    
    for i in [2,3,4,5,6,7]:
    	scores=evaluate_algorithm(train,algorithm,n_folds,i)
    	moy = sum(scores)/len(scores)
    	k_scores.append(moy)
    k = k_scores.index(max(k_scores)) + 2
    print('accuracy : %.3f%%' %(max(k_scores) *100) )
    return k
	

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors


#pour chaque ligne de tests recuperer les k plus proches voisins
def get_all_neighbors(train,test,num_neighbors):
	distances = list()
	for test_row in test:
		dist = get_neighbors(train,test_row,num_neighbors)
		distances.append(dist)
	return distances

#classifier les lignes de tests 
def predict_classification(train, test, num_neighbors):
	neighbors = get_all_neighbors(train, test, num_neighbors)
	prediction = list()
	
	for row in neighbors:
		l=list()
		for i in range(len(row)):
			l.append(row[i][-1])

		#prendre la classe majoritaire si elle existe 
		#sinon prendre le plus proches voisin 
		#dans notre cas c'est le premier element de la liste
		a = l.count(0)
		b= l.count(1)
		c= l.count(2)
		d=l.count(3)
		maxi = max(a,b,c,d)

		if maxi == a:
			prediction.append(0)
		else:
			if maxi == b:
				prediction.append(1)
			else: 
				if maxi==c :
					prediction.append(2)
				else:
					prediction.append(3)
		
	return prediction

#-----------------------------------------------------------------------------------------------------------------------------------------
l = load_csv('train.csv')
for i in range(21):
	if i==2 or i==7:
		str_column_to_float(l,i)
	else :	
		str_column_to_int(l,i)	
	

(test,train)=cross_validation_split1(l,4)
k = best_k(train,predict_classification,5)
print ('best k = ',k)
print(predict_classification(train,test,k))
