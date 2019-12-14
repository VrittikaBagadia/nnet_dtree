import pandas as pd
import numpy as np
import math
from datetime import datetime
import sys
from sklearn import tree, metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sys

def calc_entropy(target_attribute_column):
	uniqueValues, occurCount = np.unique(target_attribute_column, return_counts=True)
	total = len(target_attribute_column)
	if (total != sum(occurCount)):
		print("some problem ")
	entropy = 0
	for count in occurCount:
		entropy += -1 * (count/total) * math.log(count/total)
	return (entropy)

def calc_IG(data, feature, target_attribute):
	# print(feature)
	entropy_parent = calc_entropy(data[:,target_attribute])
	# split on feature
	classes, counts = np.unique(data[:,feature], return_counts=True)
	total = sum(counts)
	entropy_child = 0
	for i in range(len(classes)):
	# for distinct_class in classes:
		distinct_class = classes[i]
		sub_df = []
		for j in range(len(data)):
			if (data[j][feature] == distinct_class):
				sub_df.append(list(data[j]))
		sub_df = np.array(sub_df)
		# sub_df = data.loc[data[feature] == distinct_class]
		temp_entropy = calc_entropy(sub_df[:,target_attribute])
		entropy_child += ((counts[i]/total) * temp_entropy)
	ans = entropy_parent - entropy_child
	return (ans)

def calc_IG_partC(data, feature, target_attribute, continuous_columns):
	entropy_parent = calc_entropy(data[:,target_attribute])

	# if feature is continuous, calculate median and split accordingly
	if (feature in continuous_columns):
		count0 = 0
		sub_df_0 = []
		count1 = 0
		sub_df_1 = []
		median = np.median(data[:,feature])
		for i in range(len(data)):
			if (data[i][feature] <= median):
				count0 += 1
				sub_df_0.append(list(data[i]))
			else:
				count1 += 1
				sub_df_1.append(list(data[i]))
		sub_df_0 = np.array(sub_df_0)
		sub_df_1 = np.array(sub_df_1)
		temp_entropy0 = calc_entropy(sub_df_0[:,target_attribute])
		if (len(sub_df_1) == 0):
			temp_entropy1 = 0
		else:
			temp_entropy1 = calc_entropy(sub_df_1[:,target_attribute])
		entropy_child = (count0*temp_entropy0 + count1*temp_entropy1)/(count0 + count1)
		ans = entropy_parent - entropy_child
	else:
		classes, counts = np.unique(data[:,feature], return_counts=True)
		total = sum(counts)
		entropy_child = 0

		length = len(classes)
		sub_df = [[] for i in range(length)]
		for i in range(len(data)):
			for index in range(length):
				if (classes[index] == data[i][feature]):
					break
			sub_df[index].append(list(data[i]))
		for i in range(length):
			d = np.array(sub_df[i])
			if (len(d) == 0):
				temp_entropy = 0
			else:
				temp_entropy = calc_entropy(d[:,target_attribute])
			entropy_child += (counts[i]/total)*temp_entropy
		ans = entropy_parent - entropy_child
	return (ans)

def run(data, features, target_attribute):
	if (len(data) == 0):		# khaali dataframe
		print("aisa kaise ")

	elif (len(np.unique(data[:,target_attribute])) == 1): 		# pure leaf
		return (['leaf',-1,(data[0][target_attribute])], 1)#correct label

	elif (len(features) == 0):	  # impure leaf but no feature to split on - majority prediction
		values, counts = np.unique(data[:,target_attribute], return_counts = True)
		if (counts[0] > counts[1]):
			return (['leaf',-1,(values[0])],1)
		else:
			return (['leaf',-1,(values[1])],1)

	else:
		# split on next best feature
		IGs = [calc_IG(data, feature, target_attribute) for feature in features]
		# print(IGs)
		maxi = max(IGs)
		index_max = IGs.index(maxi)
		# IGs = np.array(IGs)
		# index_max = np.argmax(IGs)
		best_feature = features[index_max]
		# print('best feature: ' + str(best_feature))
		values, counts = np.unique(data[:,target_attribute], return_counts = True)
		if (counts[0] > counts[1]):
			majority = values[0]
		else:
			majority = values[1]
		if (maxi<=0):
			return (['leaf',-1,majority],1)

		tree = ['internal',best_feature, majority, {}]
		num_nodes = 1
		# tree = {best_feature:{}}

		new_features = []
		for feature in features:
			if (feature != best_feature):
				new_features.append(feature)
		classes = np.unique(data[:,best_feature], return_counts = False)
		if (len(classes) == 0):
			print('WHY ')
		for classi in classes:
			subdata = []
			for i in range(len(data)):
				if (data[i][best_feature] == classi):
					subdata.append(list(data[i]))
			subdata = np.array(subdata)
			# subdata = data.loc[data[best_feature] == classi].reset_index(drop = True)
			branch = run(subdata, new_features, target_attribute)
			# tree[best_feature][classi] = branch
			tree[3][classi] = branch[0]
			num_nodes += branch[1]

	return (tree, num_nodes)

def run_partC(data, features, target_attribute, continuous_columns):
	if (len(data) == 0):		# khaali dataframe
		print("aisa kaise ")

	elif (len(np.unique(data[:,target_attribute])) == 1): 		# pure leaf
		return (['leaf',-1,(data[0][target_attribute]), {}, -1], 1) 	# correct label

	elif (len(features) == 0):	  # impure leaf but no feature to split on - majority prediction
		values, counts = np.unique(data[:,target_attribute], return_counts = True)
		if (counts[0] > counts[1]):
			return (['leaf',-1,(values[0]), {}, -1],1)
		else:
			return (['leaf',-1,(values[1]), {}, -1],1)

	else:
		# split on next best feature
		IGs = [calc_IG_partC(data, feature, target_attribute, continuous_columns) for feature in features]
		maxi = max(IGs)
		index_max = IGs.index(maxi)
		best_feature = features[index_max]
		median = -1
		if (best_feature in continuous_columns):
			median = np.median(data[:,best_feature])

		values, counts = np.unique(data[:,target_attribute], return_counts = True)
		if (counts[0] > counts[1]):
			majority = values[0]
		else:
			majority = values[1]
		if (maxi<=0):
			return (['leaf',-1,majority, {}, -1],1)

		tree = ['internal', best_feature, majority, {}, median]
		num_nodes = 1

		new_features = []
		for feature in features:
			if (feature != best_feature):
				new_features.append(feature)

		if (best_feature in continuous_columns):
			classes = [0,1]
			subdata0 = []
			subdata1 = []
			for i in range(len(data)):
				if (data[i][best_feature] <= median):
					subdata0.append(list(data[i]))
				else:
					subdata1.append(list(data[i]))
			# subdata0 = subdata0.values
			# subdata1 = subdata1.values
			subdata0 = np.array(subdata0)
			subdata1 = np.array(subdata1)
			branch0 = run_partC(subdata0, features, target_attribute, continuous_columns)
			branch1 = run_partC(subdata1, features, target_attribute, continuous_columns)
			tree[3][0] = branch0[0]
			tree[3][1] = branch1[0]
			num_nodes += branch0[1] + branch1[1]
		else:
			classes = np.unique(data[:,best_feature], return_counts = False)
			if (len(classes) == 0):
				print('WHY ')
			for classi in classes:
				subdata = []
				for i in range(len(data)):
					if (data[i][best_feature] == classi):
						subdata.append(list(data[i]))
				# subdata = subdata.values
				subdata = np.array(subdata)
				# subdata = data.loc[data[best_feature] == classi].reset_index(drop = True)
				branch = run_partC(subdata, new_features, target_attribute, continuous_columns)
				# tree[best_feature][classi] = branch
				tree[3][classi] = branch[0]
				num_nodes += branch[1]

	return (tree, num_nodes)

def find_feature(tree, feature):
	if (tree[0] == 'leaf'):
		return ([])
	if (tree[1] == feature):
		li = [tree[4]]
	else:
		li = []
	maxi = []
	for value in tree[3].keys():
		temp = find_feature(tree[3][value], feature)
		if (len(temp) > len(maxi)):
			maxi = temp
	li = li + maxi
	return li

def find_features(tree):
	start = datetime.now()
	continuous_columns_1 = [1, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
	continuous_columns = [(col-1) for col in continuous_columns_1]

	alli = []
	for feature in continuous_columns:
		x = find_feature(tree, feature)
		print(str(feature+1) + " : " + str(len(x)) + " times at thresholds:")
		print(x)
		alli.append(x)
	return alli

def count_nodes(tree):
	if (tree[0] == 'leaf'):
		return 1
	res = 1
	for value in tree[3].keys():
		res += count_nodes(tree[3][value])
	return(res)

def predict_example(eg, features, tree):
	if (tree[0] == 'leaf'):			# leaf node	
		return tree[2]
	value = eg[tree[1]]
	if (value in tree[3].keys()):
		return ( predict_example(eg, features, tree[3][value]) )
	return (tree[2])

def predict_example_partC(eg, features, tree, continuous_columns):
	if (tree[0] == 'leaf'):			# leaf node	
		return tree[2]
	feature = tree[1]
	if (feature in continuous_columns):
		median = tree[4]
		if (eg[feature] <= median):
			value = 0
		else:
			value = 1
	else:
		value = eg[feature]
	if (value in tree[3].keys()):
		return ( predict_example_partC(eg, features, tree[3][value],continuous_columns) )
	return (tree[2])

def predict(dataset, features, tree, target_attribute):
	correct = dataset[:,target_attribute]
	predicted = np.apply_along_axis(predict_example, 1, dataset, features, tree)
	return (1 - np.count_nonzero(correct-predicted)/len(correct))

def predict_partC(dataset, features, tree, target_attribute, continuous_columns):
	correct = dataset[:,target_attribute]
	predicted = np.apply_along_axis(predict_example_partC, 1, dataset, features, tree, continuous_columns)
	return (1 - np.count_nonzero(correct-predicted)/len(correct))

def read_training(filename):
	df = pd.read_csv(filename, skiprows = [1])
	df = df.drop(columns = ['X0'])
	df = df.values
	continuous_columns = [1, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
	medians = []
	for col in continuous_columns:
		col = col-1
		median = np.median(df[:,col])
		medians.append(median)
		df[:,col] = np.where(df[:,col] > median, 1, 0)
	return(df, medians)

def read(filename, medians):
	df = pd.read_csv(filename, skiprows = [1])
	df = df.drop(columns = ['X0'])
	df = df.values
	continuous_columns = [1, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
	for i in range(14):
		col = continuous_columns[i] - 1
		median = medians[i]
		df[:,col] = np.where(df[:,col] > median, 1, 0)
	return(df)

def read_normal(filename):
	df = pd.read_csv(filename, skiprows = [1])
	df = df.drop(columns = ['X0'])
	df = df.values 			# df to np.array
	return df

def read_1_hot_encoding(filename):
	categorical_features = [2,3,5,6,7,8,9,10]
	df = pd.read_csv(filename, skiprows = [1])
	df = df.drop(columns = ['X0'])
	df = df.values
	m = len(df)
	# print(len(df[0]))

	final_df = df[:,0:2]
	arr = df[:,2]
	temp_df = np.zeros((m, 7))
	temp_df[np.arange(m), arr] = 1
	final_df = np.concatenate((final_df, temp_df), axis = 1)

	arr = df[:,3]
	temp_df = np.zeros((m, 4))
	temp_df[np.arange(m), arr] = 1
	final_df = np.concatenate((final_df, temp_df), axis = 1)

	temp_df = [[val] for val in df[:,4]]
	final_df = np.concatenate((final_df, temp_df), axis = 1)

	for index in [5,6,7,8,9,10]:
		temp_df = np.zeros((m, 12))
		arr = df[:,index]
		arr = arr+2
		temp_df[np.arange(m), arr] = 1
		final_df = np.concatenate((final_df, temp_df), axis =1 )
	final_df = np.concatenate((final_df, df[:,11:]), axis = 1)

	# print(len(final_df[0]))
	return (final_df)


	# for index in range(1, len(df[0]) - 1):
	# 	if (index not in categorical_features):
	# 		arr2 = np.array([[val] for val in df[:,index]])
	# 		final_df = np.concatenate((final_df, arr2), axis = 1)
	# 	else:
	# 		arr = df[:,index]
	# 		mini = arr.min()
	# 		arr = arr + mini
	# 		temp_df = np.zeros((m, arr.max()+1))
	# 		temp_df[np.arange(m),arr] = 1
	# 		final_df = np.concatenate((final_df, temp_df), axis=1)
	# # one_hot_encoded_df = pd.get_dummies(df)
	# # one_hot_encoded_df = one_hot_encoded_df.values
	
def pruning_node(node, root, val_data, features, target_attribute):
	if (node[0] == 'leaf'):
		return(-1, node)
	node[0] = 'leaf'
	acc = predict(val_data, features, root, target_attribute)
	node[0] = 'internal'
	max_acc = acc
	max_acc_node = node

	for value in node[3].keys():
		temp = pruning_node(node[3][value], root, val_data, features, target_attribute)
		if (max_acc_node == root):
			max_acc = temp[0]
			max_acc_node = temp[1]
		elif (temp[0] >= max_acc):
			max_acc = temp[0]
			max_acc_node = temp[1]
	return(max_acc, max_acc_node)

def prune(tree, val_data, features, target_attribute, training_data, test_data):
	print("start pruning ")
	count = 0
	stop = 0
	training_accuracies = []
	validation_accuracies = []
	test_accuracies = []
	num_nodes = []
	while (stop == 0):
		node_temp = tree
		# s = datetime.now()
		old_acc = predict(val_data, features, tree, target_attribute)
		training_acc = predict(training_data, features, tree, target_attribute)
		test_acc = predict(test_data, features, tree, target_attribute)
		num = count_nodes(tree)
		training_accuracies.append(training_acc)
		validation_accuracies.append(old_acc)
		test_accuracies.append(test_acc)
		num_nodes.append(num)
		# print("time for prediction " + str(datetime.now() - s) )
		starti = datetime.now()
		(acc, node) = pruning_node(node_temp, tree, val_data, features, target_attribute)
		print(datetime.now() - starti)
		if (acc < old_acc):
			stop = 1
		else:
			if (node[0] == 'leaf'):
				print("what does this mean ")
			node[0] = 'leaf'
			count+=1
			print(str(count) + "accuracy improved to " + str(acc))

		# if (node[0] == 'leaf'):
		# 	print("what does this mean")
		# node[0] = 'leaf'
		# count+=1
		# new_acc = acc
		# new_acc = predict(val_data, features, tree, target_attribute)
		# print(str(count) + "accuracy improved to " + str(new_acc))
		# if (old_acc > new_acc):
		# 	stop = 1
			# node[0] = 'internal'
	return (num_nodes, training_accuracies, test_accuracies, validation_accuracies)

def make_all_leaves(tree):
	if (tree[0] == 'internal'):
		tree[0] = 'leaf'
		for value in tree[3].keys():
			make_all_leaves(tree[3][value])

def all_leaves_accuracy(node, tree, training_data, test_data, validation_data, features, target_attribute, part):
	if (node[1] == -1):			# was a leaf
		return ([],[],[],[])
	node[0] = 'internal'
	print(node[1])
	if (part == 'C'):
		continuous_columns_1 = [1, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
		continuous_columns = [(col-1) for col in continuous_columns_1]
		acc1 = predict_partC(training_data, features, tree, target_attribute, continuous_columns)
		acc2 = predict_partC(test_data, features, tree, target_attribute, continuous_columns)
		acc3 = predict_partC(validation_data, features, tree, target_attribute, continuous_columns)
	else:
		acc1 = predict(training_data, features, tree, target_attribute)
		acc2 = predict(test_data, features, tree, target_attribute)
		acc3 = predict(validation_data, features, tree, target_attribute)
	num_nodes = [len(node[3].keys())]		# number of nodes added by making current node an internal node	
	training_accuracies = [acc1]
	test_accuracies = [acc2]
	validation_accuracies = [acc3]

	for value in node[3].keys():
		num, tr, te, va = all_leaves_accuracy(node[3][value], tree, training_data, test_data, validation_data, features, target_attribute, part)
		to_add = num_nodes[-1]
		for ai in num:
			num_nodes.append(ai+to_add)
		training_accuracies = training_accuracies + tr
		test_accuracies = test_accuracies + te
		validation_accuracies = validation_accuracies + va
	return (num_nodes, training_accuracies, test_accuracies, validation_accuracies)

def finding_nodewise_accuracy(tree, training_data, testing_data, validation_data, features, target_attribute, part):
	make_all_leaves(tree)
	print("converted all nodes to leaves ")
	# acc1 = predict(training_data, features, tree, target_attribute)
	# acc2 = predict(testing_data, features, tree, target_attribute)
	# acc3 = predict(validation_data, features, tree, target_attribute)
	# num_nodes = [1]			# only root added
	# training_accuracies = [acc1]
	# test_accuracies = [acc2]
	# validation_accuracies = [acc3]
	num, tr, te, va = all_leaves_accuracy(tree, tree, training_data, testing_data, validation_data, features, target_attribute, part)
	for i in range(len(num)):
		num[i] += 1
	# training_accuracies = training_accuracies + tr
	# test_accuracies = test_accuracies + te
	# validation_accuracies = validation_accuracies + va

	return (num, tr, te, va)

def plot(num_nodes, training_accuracies, test_accuracies, validation_accuracies, part):
	plt.title("Accuracy vs Number of nodes")
	# plt.scatter(X,Y)
	plt.plot(num_nodes,training_accuracies,'r-', label = 'Training set')
	plt.plot(num_nodes,test_accuracies,'g-', label = 'Test set')
	plt.plot(num_nodes,validation_accuracies,'b-', label = 'Validation set')
	if (part == 'B'):
		plt.xlim(max(num_nodes), min(num_nodes))
	else:
		plt.xlim(0, max(num_nodes))
	plt.xlabel('number of nodes')
	plt.ylabel('accuracy')
	plt.legend()
	plt.savefig('accuracy_variation_part' + part + '.png')
	plt.show()

def partA(training_file, validation_file, test_file):
	target_attribute = 23
	all_features = [i for i in range(23)]
	all_features_copy = list(all_features)
	# read training data
	df, medians = read_training(training_file)

	start = datetime.now()
	tree , num_nodes = run(df, all_features, target_attribute)
	# print("number of nodes " + str(num_nodes))
	print("number of nodes " + str(count_nodes(tree)))
	print ('training time : ' + str(datetime.now() - start))

	# prediction on training data
	acc1 = predict(df, all_features_copy, tree, target_attribute)
	print("training data accuracy " + str(acc1))

	# prediction on test data
	test_df = read(test_file, medians)
	acc2 = predict(test_df, all_features_copy, tree, target_attribute)
	print("test data accuracy " + str(acc2))

	# prediction on validation data
	val_df = read(validation_file, medians)
	acc3 = predict(val_df, all_features_copy, tree, target_attribute)
	print("validation data accuracy " + str(acc3))

	# start = datetime.now()
	# num_nodes, training_accuracies, test_accuracies, validation_accuracies = finding_nodewise_accuracy(tree, df, test_df, val_df, all_features_copy, target_attribute, 'A')
	# print('time taken for finding accuracy nodewise: ' + str(datetime.now() - start))

	# plot(num_nodes, training_accuracies, test_accuracies, validation_accuracies, 'A')
	# return num_nodes, training_accuracies, test_accuracies, validation_accuracies
	return [], [], [], []

def partB(training_file, validation_file, test_file):
	target_attribute = 23
	all_features = [i for i in range(23)]
	all_features_copy = list(all_features)
	# read training data
	df, medians = read_training(training_file)

	start = datetime.now()
	tree , num_nodes = run(df, all_features, target_attribute)
	print("number of nodes " + str(num_nodes))
	print("number of internal nodes " + str(count_nodes(tree)))
	print ('training time : ' + str(datetime.now() - start))

	# prediction on training data
	acc1 = predict(df, all_features_copy, tree, target_attribute)
	print("training data accuracy " + str(acc1))

	# prediction on test data
	test_df = read(test_file, medians)
	acc2 = predict(test_df, all_features_copy, tree, target_attribute)
	print("test data accuracy " + str(acc2))

	# prediction on validation data
	val_df = read(validation_file, medians)
	acc3 = predict(val_df, all_features_copy, tree, target_attribute)
	print("validation data accuracy " + str(acc3))


	print (str(datetime.now() - start))
	num_nodes, training_accuracies, test_accuracies, validation_accuracies = prune(tree, val_df, all_features_copy, target_attribute, df, test_df)
	print (str(datetime.now() - start))


	print('tree has been pruned')
	acc1 = predict(df, all_features_copy, tree, target_attribute)
	print("training data accuracy " + str(acc1))

	acc2 = predict(test_df, all_features_copy, tree, target_attribute)
	print("test data accuracy " + str(acc2))

	acc3 = predict(val_df, all_features_copy, tree, target_attribute)
	print("validation data accuracy " + str(acc3))

	plot(num_nodes, training_accuracies, test_accuracies, validation_accuracies,'B')
	return (num_nodes, training_accuracies, test_accuracies, validation_accuracies)

def partC(training_file, validation_file, test_file):
	target_attribute = 23
	all_features = [i for i in range(23)]
	all_features_copy = list(all_features)		# memcpy type

	continuous_columns_1 = [1, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
	continuous_columns = [(col-1) for col in continuous_columns_1]

	# read data
	df = read_normal('data/credit-cards.train.csv')
	test_df = read_normal('data/credit-cards.test.csv')
	val_df = read_normal("data/credit-cards.val.csv")

	start = datetime.now()
	tree , num_nodes = run_partC(df, all_features, target_attribute, continuous_columns)
	print("number of nodes " + str(num_nodes))
	print("number of internal nodes " + str(count_nodes(tree)))
	print ('training time : ' + str(datetime.now() - start))

	# prediction on training data
	acc1 = predict_partC(df, all_features_copy, tree, target_attribute, continuous_columns)
	print("training data accuracy " + str(acc1))

	# prediction on test data
	acc2 = predict_partC(test_df, all_features_copy, tree, target_attribute, continuous_columns)
	print("test data accuracy " + str(acc2))

	# prediction on validation data
	acc3 = predict_partC(val_df, all_features_copy, tree, target_attribute, continuous_columns)
	print("validation data accuracy " + str(acc3))

	features_freq = find_features(tree)

	# start = datetime.now()
	# num_nodes, training_accuracies, test_accuracies, validation_accuracies = finding_nodewise_accuracy(tree, df, test_df, val_df, all_features_copy, target_attribute, 'C')
	# print('time taken for finding accuracy nodewise: ' + str(datetime.now() - start))
	# plot(num_nodes, training_accuracies, test_accuracies, validation_accuracies, 'C')
	# return num_nodes, training_accuracies, test_accuracies, validation_accuracies, features_freq
	return [], [], [], [], features_freq

def partD(training_file, validation_file, test_file):
	# read data
	df = read_normal(training_file)
	test_df = read_normal(test_file)
	val_df = read_normal(validation_file)

	clf = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth = 6, min_samples_split = 0.001, min_samples_leaf = 0.01, max_features = 0.65)
	clf = clf.fit(df[:,0:-1], df[:,-1])
	prediction_training = clf.predict(df[:,0:-1])
	prediction_test = clf.predict(test_df[:,0:-1])
	prediction_validation = clf.predict(val_df[:,0:-1])

	correct_training = df[:,-1]
	correct_test = test_df[:,-1]
	correct_validation = val_df[:,-1]

	# acc_training = (1 - np.count_nonzero(correct_training-prediction_training )/len(correct_training))
	acc_training = metrics.accuracy_score(correct_training, prediction_training)
	acc_test = metrics.accuracy_score(correct_test, prediction_test)
	acc_validation = metrics.accuracy_score(correct_validation, prediction_validation)

	s_test = metrics.f1_score(correct_test, prediction_test)

	print("training accuracy : " + str(acc_training))
	print("validation accuracy : " + str(acc_validation))
	print("test accuracy : " + str(acc_test))
	# print(s_test)

def partE(training_file, validation_file, test_file):
	df = read_1_hot_encoding(training_file)
	test_df = read_1_hot_encoding(test_file)
	val_df = read_1_hot_encoding(validation_file)
	# print(len(df[0]))
	clf = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth = 6, min_samples_split = 0.001, min_samples_leaf = 0.01, max_features = 0.65)
	clf = clf.fit(df[:,0:-1], df[:,-1])
	prediction_training = clf.predict(df[:,0:-1])
	prediction_test = clf.predict(test_df[:,0:-1])
	prediction_validation = clf.predict(val_df[:,0:-1])

	correct_training = df[:,-1]
	correct_test = test_df[:,-1]
	correct_validation = val_df[:,-1]

	# acc_training = (1 - np.count_nonzero(correct_training-prediction_training )/len(correct_training))
	acc_training = metrics.accuracy_score(correct_training, prediction_training)
	acc_test = metrics.accuracy_score(correct_test, prediction_test)
	acc_validation = metrics.accuracy_score(correct_validation, prediction_validation)

	# s_test = metrics.f1_score(correct_test, prediction_test)

	print("training accuracy : " + str(acc_training))
	print("validation accuracy : " + str(acc_validation))
	print("test accuracy : " + str(acc_test))
	# print(s_test)

def partF(training_file, validatin_file, test_file):
	df = read_1_hot_encoding(training_file)
	test_df = read_1_hot_encoding(test_file)
	val_df = read_1_hot_encoding(validation_file)
	clf = RandomForestClassifier(criterion = "entropy", n_estimators=23, bootstrap = True, max_features = 10)
	clf.fit(df[:,:-1], df[:,-1])
	acc_training = clf.score(df[:,:-1], df[:,-1])
	acc_validation = clf.score(val_df[:,:-1], val_df[:,-1])
	acc_test = clf.score(test_df[:,:-1], test_df[:,-1])
	print("training accuracy : " + str(acc_training))
	print("validation accuracy : " + str(acc_validation))
	print("test accuracy : " + str(acc_test))
	

training_file = sys.argv[2]
test_file = sys.argv[3]
validation_file = sys.argv[4]

if (sys.argv[1] == '1'):
	num_nodes, training_accuracies, test_accuracies, validation_accuracies = partA(training_file, validation_file, test_file)
elif (sys.argv[1] == '2'):
	num_nodes, training_accuracies, test_accuracies, validation_accuracies = partB(training_file, validation_file, test_file)
elif (sys.argv[1] == '3'):
	num_nodes, training_accuracies, test_accuracies, validation_accuracies, features_frequency = partC(training_file, validation_file, test_file)
elif (sys.argv[1] == '4'):
	partD(training_file, validation_file, test_file)
elif (sys.argv[1] == '5'):
	partE(training_file, validation_file, test_file)
elif (sys.argv[1] == '6'):
	partF(training_file, validation_file, test_file)
else:
	print("invalid input")

	
# training_file = 'data/credit-cards.train.csv'
# validation_file = "data/credit-cards.val.csv"
# test_file = 'data/credit-cards.test.csv'

# num_nodes, training_accuracies, test_accuracies, validation_accuracies = partA(training_file, validation_file, test_file)
# num_nodes, training_accuracies, test_accuracies, validation_accuracies = partB(training_file, validation_file, test_file)
# num_nodes, training_accuracies, test_accuracies, validation_accuracies, features_frequency = partC(training_file, validation_file, test_file)
# partD(training_file, validation_file, test_file)
# partE(training_file, validation_file, test_file)
# partF(training_file, validation_file, test_file)

