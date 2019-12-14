import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import time
import sys

def read(filename):
	start = datetime.now()
	df = pd.read_csv(filename, header = None)
	df = df.values
	m = len(df)

	arr = df[:,df.shape[1]-1]
	final_df = np.zeros((m,10))
	final_df[np.arange(m),arr] = 1
	for index in range(len(df[0]) - 1):
		arr = df[:,index]
		arr = arr-1
		if (index % 2 == 0):
			maxi = 4
		else:
			maxi = 13
		temp_df = np.zeros((m, maxi))
		temp_df[np.arange(m),arr] = 1
		final_df = np.concatenate((final_df, temp_df), axis=1)

	print(datetime.now() - start)
	x = final_df[:,10:]
	y = final_df[:,:10]
	return x,y

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
def relu(x):
	return np.maximum(0,x)
class NeuralNetwork:
	def __init__(self, batch_size, num_inputs, num_outputs, hidden_layers, learning_rate, activation_func, non_linearity = "sigmoid", variable_learning_rate = 0, tol = 0.0001):
		print(hidden_layers)
		self.batch_size = batch_size
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.layers_size = [num_inputs] + hidden_layers + [num_outputs]
		self.num_layers = len(hidden_layers) + 1
		self.w = [(np.random.rand(self.layers_size[i+1],self.layers_size[i]+1) * 2 - 1)/100 for i in range(self.num_layers)]
		self.learning_rate = learning_rate
		self.activation_func = activation_func
		self.non_linearity = non_linearity
		self.variable_learning_rate = variable_learning_rate
		self.tol = tol

	def calc_loss(self, X, Y):
		ans = X.T
		for i in range(self.num_layers-1):
			ans = self.activation_func(np.dot(self.w[i], np.insert(ans,0,1,axis = 0)))
		ans = sigmoid(np.dot(self.w[self.num_layers - 1], np.insert(ans,0,1,axis = 0)))
		ans = ans.T
		return (( np.sum((ans - Y)**2) / len(Y)))

	def calc_accuracy(self,X,Y):
		ans = X.T
		for i in range(self.num_layers-1):
			ans = self.activation_func(np.dot(self.w[i], np.insert(ans,0,1,axis = 0)))
		ans = sigmoid(np.dot(self.w[self.num_layers - 1], np.insert(ans,0,1,axis = 0)))
		predicted = np.argmax(ans, axis = 0)
		actual = np.argmax(Y, axis = 1)		# row wise max
		accuracy =1 - (np.count_nonzero(actual-predicted)/float(len(actual)))
		return accuracy*100

	def test(self,X,Y):
		ans = X.T
		for i in range(self.num_layers-1):
			ans = self.activation_func(np.dot(self.w[i], np.insert(ans,0,1,axis = 0)))
		ans = sigmoid(np.dot(self.w[self.num_layers - 1], np.insert(ans,0,1,axis = 0)))
		predicted = np.argmax(ans, axis = 0)
		actual = np.argmax(Y, axis = 1)		# row wise max
		return predicted,actual
		# accuracy =1 - (np.count_nonzero(actual-predicted)/float(len(actual)))
		# accuracy = accuracy * 100
		# labels = [0,1,2,3,4,5,6,7,8,9]
		# cm = confusion_matrix(actual, predicted, labels)
		# return accuracy,cm

	def train(self,X,Y):
		start = time.time()
		m = X.shape[0]
		
		epoch_number = 0
		stop = 0
		new_loss = self.calc_loss(X,Y)
		# for epoch_number in range(1000):
		while (epoch_number < 100 and stop == 0):
			print("epoch: " + str(epoch_number))
			print("accuracy: " + str(self.calc_accuracy(X,Y)))
			epoch_number+=1

			current_loss = new_loss
			# print("old loss : " + str(current_loss))

			# divide into batches and each batch performs one update
			if (m%self.batch_size == 0):
				num_batches = m/self.batch_size
			else:
				num_batches = m/self.batch_size + 1
			for batch_num in range(num_batches):
				X_working = X[batch_num*self.batch_size: min(m,(batch_num+1)*self.batch_size), :]
				Y_working = Y[batch_num*self.batch_size: min(m,(batch_num+1)*self.batch_size), :]
		
				outputs = [None for i in range(self.num_layers + 1)]		# +1 for hidden layer
				outputs[0] = X_working.T
				# FORWARD PROPOGATION
				if (self.non_linearity == 'sigmoid'):
					for i in range(self.num_layers):
						outputs[i+1] = self.activation_func(np.dot(self.w[i], np.insert(outputs[i],0,1,axis = 0)))
				else:
					for i in range(self.num_layers-1):
						outputs[i+1] = self.activation_func(np.dot(self.w[i], np.insert(outputs[i],0,1,axis = 0)))
					outputs[self.num_layers] = sigmoid(np.dot(self.w[self.num_layers-1], np.insert(outputs[self.num_layers-1],0,1,axis = 0)))

				# BACK PROPOGATION
				errors = [None for i in range(self.num_layers)]
				errors[self.num_layers-1] = np.multiply((outputs[self.num_layers] - Y_working.T) , np.multiply(outputs[self.num_layers],(1 - outputs[self.num_layers])))
				for i in range(self.num_layers-2, -1, -1):
					if (self.non_linearity == 'sigmoid'):
						errors[i] = np.multiply(np.dot(self.w[i+1].T[1:], errors[i+1]), np.multiply(outputs[i+1],(1-outputs[i+1])))
					else:
						errors[i] = np.multiply(np.dot(self.w[i+1].T[1:], errors[i+1]), np.where(outputs[i+1] > 0, 1, 0))
				gradients = [None for i in range(self.num_layers)]
				for i in range(self.num_layers-1, -1, -1):
					gradients[i] = np.dot(errors[i], np.insert(outputs[i].T, 0, 1, axis = 1))
					self.w[i] = self.w[i] - self.learning_rate * gradients[i]
			new_loss = self.calc_loss(X,Y)
			# print("new loss : " + str(new_loss))

			if (self.variable_learning_rate == 1 and epoch_number != 1):
				if ((current_loss - new_loss) < 0.0001):
					self.learning_rate = self.learning_rate/5
					print("changed learning rate to : " + str(self.learning_rate))

			if (abs(new_loss - current_loss) < 0.000001):		# 131 - 92.13%  &   414 - 92.255%
				stop = 1
		end = time.time()
		return (end - start)

def plot_confusion_matrix(cm, title, part):
	labels = [0,1,2,3,4,5,6,7,8,9]
	df_cm = pd.DataFrame(cm, index=labels, columns = labels)
	plt.figure(figsize = (10,10))
	ax = plt.gca()
	plt.title('CONFUSION MATRIX : #units- ' + str(title))
	sn.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
	plt.yticks(rotation=0) 
	ax.hlines(labels+[10], *ax.get_xlim())
	ax.vlines(labels+[10], *ax.get_ylim())
	# plt.show()
	plt.savefig('plots' + part + '/confusion_matrix_' + str(title) + '.png')

def plot(X, Y, title, part):
	plt.title(title)
	# plt.scatter(X,Y)
	plt.plot(X,Y,'r-')
	plt.xlabel('number of hidden layer units')
	plt.ylabel(title)
	plt.savefig('plots' + part + '/' + title + '.png')
	plt.show()

def partA(training_file, test_file, target_training, target_test):
	x_train, y_train = read("data/poker-hand-training-true.data")
	x_test, y_test = read("data/poker-hand-testing.data")
	train = np.concatenate((x_train,y_train), axis = 1)
	test = np.concatenate((x_test, y_test), axis = 1)
	np.savetxt(target_training, train.astype(int), fmt='%i', delimiter=",")
	np.savetxt(target_test, test.astype(int), fmt='%i', delimiter=",")

def partB(config_file, encoded_training_file, encoded_test_file):
	f = open(config_file, 'r')
	f2 = open(encoded_training_file, 'r')
	f3 = open(encoded_test_file, 'r')
	input_size = int(f.readline().strip())
	output_size = int(f.readline().strip())
	batch_size = int(f.readline().strip())
	num_layers = int(f.readline().strip())
	s = f.readline().strip().split()
	layers = []
	for i in range(num_layers):
		layers.append(int(s[i]))
	non_linearity = f.readline().strip()
	if (non_linearity == 'sigmoid'):
		print('activation_func - sigmoid')
		function = sigmoid
	else:
		print('activation_func - relu')
		function = relu
	s = f.readline().strip()
	if (s=='fixed'):
		print('fixed learning rate')
		variable_learning_rate = 0
	else:
		print('adaptive learning rate')
		variable_learning_rate = 1

	training_data = np.array([map(int,i.strip().split(",")) for i in f2.readlines()])
	x_train = training_data[:,:85]
	y_train = training_data[:,-10:]

	NN = NeuralNetwork(batch_size,input_size,output_size,layers,0.1,function,non_linearity,variable_learning_rate)
	start = time.time()
	NN.train(x_train, y_train)
	end = time.time()
	print('training done: ')

	print('loading test data...')
	test_data = np.array([map(int,i.strip().split(",")) for i in f3.readlines()])
	x_test = test_data[:,:85]
	y_test = test_data[:,-10:]
	print('loading done')

	predicted_train, actual_train = NN.test(x_train, y_train)
	predicted_test, actual_test = NN.test(x_test, y_test)
	acc_train =1 - (np.count_nonzero(actual_train-predicted_train)/float(len(actual_train)))
	acc_test =1 - (np.count_nonzero(actual_test-predicted_test)/float(len(actual_test)))

	labels = [0,1,2,3,4,5,6,7,8,9]
	cm = confusion_matrix(actual_test, predicted_test, labels)
	print("model training time : " + str(end-start))
	print("training set accuracy : " + str(acc_train))
	print("test set accuracy : " + str(acc_test))
	print("confusion matrix ")
	print(cm)
	return acc_train, acc_test, cm

def partCandD(training_file, test_file, part, E=0, F=0):
	# single layer
	x_train, y_train = read(training_file)
	x_test, y_test = read(test_file)
	if (part == 'C'):
		# size_of_layer = [25]
		size_of_layer = [[5],[10],[15],[20],[25]]
	elif (part == 'D'):
		size_of_layer = [[5,5],[10,10],[15,15],[20,20],[25,25]]
	else:
		print("input has to be 'C' or 'D' ")
	times = []
	accuracy_train = []
	accuracy_test = []
	confusion_mat = []
	for size in size_of_layer:
		if (E == 1):
			NN = NeuralNetwork(10,85,10,size,0.1,sigmoid,'sigmoid',1)
		elif (F==1):
			NN = NeuralNetwork(10,85,10,size,0.1,relu,'relu',1)
		else:
			NN = NeuralNetwork(10,85,10,size,0.1,sigmoid,'sigmoid',0)

		temp_time = NN.train(x_train, y_train)
		times.append(temp_time)
		predicted_train, actual_train = NN.test(x_train, y_train)
		predicted_test, actual_test = NN.test(x_test, y_test)

		acc_train =1 - (np.count_nonzero(actual_train-predicted_train)/float(len(actual_train)))
		acc_test =1 - (np.count_nonzero(actual_test-predicted_test)/float(len(actual_test)))
		
		labels = [0,1,2,3,4,5,6,7,8,9]
		cm = confusion_matrix(actual_test, predicted_test, labels)

		accuracy_train.append(acc_train*100)
		accuracy_test.append(acc_test*100)
		confusion_mat.append(cm)

	print("training time : ")
	print(times)
	print("accuracy on training data: ")
	print(accuracy_train)
	print("accuracy on test data: ")
	print(accuracy_test)
	print("confusion_matrices: ")
	print(confusion_mat)

	# times = matplotlib.dates.date2num(time)
	plot(size_of_layer, times, 'training_time', part)
	plot(size_of_layer, accuracy_train, 'training_set_accuracy', part)
	plot(size_of_layer, accuracy_test, 'test_set_accuracy', part)

	for i in range(5):
		plot_confusion_matrix(confusion_mat[i], size_of_layer[i], part)
	return time, accuracy_train, accuracy_test, confusion_mat

def partE(training_file, test_file):
	time1, accuracy_train1, accuracy_test1, confusion_mat1 = partCandD(training_file, test_file, 'C', 1, 0)
	time2, accuracy_train2, accuracy_test2, confusion_mat2 = partCandD(training_file, test_file, 'D', 1, 0)
	return time1, accuracy_train1, accuracy_test1, confusion_mat1, time2, accuracy_train2, accuracy_test2, confusion_mat2

def partF(training_file, test_file):
	time1, accuracy_train1, accuracy_test1, confusion_mat1 = partCandD(training_file, test_file, 'C', 0, 1)
	time2, accuracy_train2, accuracy_test2, confusion_mat2 = partCandD(training_file, test_file, 'D', 0, 1)
	return time1, accuracy_train1, accuracy_test1, confusion_mat1, time2, accuracy_train2, accuracy_test2, confusion_mat2

if (sys.argv[1] == '0'):
	partA(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

elif (sys.argv[1] == '1'):
	acc_train, acc_test, cm = partB(sys.argv[2], sys.argv[3], sys.argv[4])

# training_file = "data/poker-hand-training-true.data"
# test_file = "data/poker-hand-testing.data"
# partA(training_file, test_file)

# config_file = 'config.txt'
# encoded_training_file = 'one_hot_encoded_data/one_hot_encoded_train.csv'
# encoded_test_file = 'one_hot_encoded_data/one_hot_encoded_test.csv'
# acc_train, acc_test = partB(config_file, encoded_training_file, encoded_test_file)

# time1, accuracy_train, accuracy_test, confusion_mat = partCandD(training_file, test_file, 'C', 0)
# time2, accuracy_train2, accuracy_test2, confusion_mat2 = partCandD(training_file, test_file, 'D', 0)
# time1, accuracy_train1, accuracy_test1, confusion_mat1, time2, accuracy_train2, accuracy_test2, confusion_mat2 = partE(training_file, test_file)
# time1, accuracy_train1, accuracy_test1, confusion_mat1, time2, accuracy_train2, accuracy_test2, confusion_mat2 = partF(training_file, test_file)
