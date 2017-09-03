import scipy.io as sio
import numpy as np 
import matplotlib.pyplot as plt 


class NumberRecognizer():
	def __init__(self, weight_decay_coefficient = 0.001, hid_num = 37, n_iter = 1000, learning_rate = 0.22, \
		momentum_multiplier = 0.9, early_stop = True, batch_size = 100):
		# weight_decay_coefficient 			L2 penalty
		# hid_num							Number of hidden units
		# n_iter 							Number of iterations
		# momentum_multiplier 				Momentum coefficent
		# early_stop						Use model that minimizes validation data
		# batch_size						Size of mini-batch
		# features_num   					each training input is a 16x16 image, so 256 features in total
		# class_num							number of output class, predict the letter to be 1 ~ 10

		self.weight_decay_coefficient = weight_decay_coefficient 	
		self.hid_num =  hid_num	
		self.n_iter = n_iter 
		self.learning_rate = learning_rate
		self.momentum_multiplier = momentum_multiplier
		self.early_stop = early_stop
		self.batch_size = batch_size
		self.features_num = 256
		self.class_num = 10

	def logistic(self,matrix):
		# calculate the logistic of a matrix
		return 1.0/(1.0+ np.exp(-matrix))

	def log_sum_exp_over_row(self,a):
		# a safe way to calculate the denominator of a soft-max function
		max_small_value = a.max(axis = 1)
		return np.log(np.sum(np.exp(a- max_small_value.reshape(-1, 1)), axis = 1)	) + max_small_value

	def model_to_theta(self,model):
		# a model is a dictionary of 'input_to_hid_weights' and 'hid_to_class_weights'
		# this function straightens out all model's parameter to a single array
		return np.append(model['input_to_hid_weights'].ravel(), model['hid_to_class_weights'].ravel())

	def theta_to_model(self,theta):
		# this function returns the model from the theta acquired from the function above
		input_to_hid_weights = theta[:self.features_num*self.hid_num].reshape(self.features_num,self.hid_num)
		hid_to_class_weights = theta[self.features_num*self.hid_num:].reshape(self.hid_num,self.class_num)
		return {'input_to_hid_weights': input_to_hid_weights, 'hid_to_class_weights': hid_to_class_weights}

	def initial_model(self):
		# this function randomly generates the initial model
		n_params = (self.features_num + self.class_num)*self.hid_num
		return self.theta_to_model(0.1*np.random.rand(n_params))

	def loss(self,model, data_input, data_output, weight_decay_coefficient):
		#this function calculates the loss of the prediction
		hid_input = data_input.dot(model['input_to_hid_weights'])
		hid_output = self.logistic(hid_input)

		class_input = hid_output.dot(model['hid_to_class_weights'])
		class_nomalizer = self.log_sum_exp_over_row(class_input)
		log_class_prob = class_input - class_nomalizer.reshape(-1, 1)
		class_prob = np.exp(log_class_prob)

		classification_loss = -np.mean(np.sum(data_output*log_class_prob, axis = 1))
		theta = self.model_to_theta(model)
		weight_decay_loss = 0.5*weight_decay_coefficient*theta.dot(theta)
		return classification_loss + weight_decay_loss

	def d_loss_d_model(self,model, data_input, data_output, weight_decay_coefficient):
		#this function calculates the derivative of the loss with respect to the weights
		hid_input = data_input.dot(model['input_to_hid_weights'])
		hid_output = self.logistic(hid_input)
		
		class_input = hid_output.dot(model['hid_to_class_weights'])
		class_nomalizer = self.log_sum_exp_over_row(class_input)
		log_class_prob = class_input - class_nomalizer.reshape(-1, 1)
		class_prob = np.exp(log_class_prob)

		diff = class_prob- data_output

		deriv_hid_to_class_weights = hid_output.T.dot(diff)/len(data_input) + self.weight_decay_coefficient*model['hid_to_class_weights']
		deriv_input_to_hid_weights = data_input.T.dot(diff.dot(model['hid_to_class_weights'].T)*hid_output*(1-hid_output))/len(data_input)+\
			self.weight_decay_coefficient*model['input_to_hid_weights']
		return {'input_to_hid_weights': deriv_input_to_hid_weights, 'hid_to_class_weights': deriv_hid_to_class_weights}

	def predict(self,model, data_input, data_output):
		# this function tests how good a model perform based on the test data
		# return the error percent
		hid_input = data_input.dot(model['input_to_hid_weights'])
		hid_output = self.logistic(hid_input)
		class_input = hid_output.dot(model['hid_to_class_weights'])
		class_nomalizer = self.log_sum_exp_over_row(class_input)
		log_class_prob = class_input - class_nomalizer.reshape(-1, 1)
		class_prob = np.exp(log_class_prob)

		predicted = class_prob.argmax(axis = 1)
		output = data_output.argmax(axis = 1)

		result = 0
		for i in range(len(output)):
			if output[i]!= predicted[i]:
				result+=1
		return result/len(output)

	def fit(self,training_input, training_output, valid_input, valid_output):
		# this function trains the model based on the training and validation data
		model = self.initial_model()
		training_case_num = len(training_input)

		theta = self.model_to_theta(model)
		momentum_speed = theta*0
		best_so_far = {}
		training_data_losses = []
		validation_data_losses = []
		
		if self.early_stop:
			best_so_far['theta'] =-1
			best_so_far['validation_loss'] = 10000
			best_so_far['after_n_iter'] = -1

		for optimization_interation_i in range(self.n_iter):
			model = self.theta_to_model(theta)

			training_batch_start = optimization_interation_i*self.batch_size%training_case_num
			training_data_mini = {'input': training_input[training_batch_start: training_batch_start+self.batch_size],\
				'output': training_output[training_batch_start: training_batch_start+ self.batch_size]}
			
			gradient = self.model_to_theta(self.d_loss_d_model(model, training_data_mini['input'], training_data_mini['output'], self.weight_decay_coefficient))
			momentum_speed = momentum_speed*self.momentum_multiplier -gradient
			theta = theta + self.learning_rate*momentum_speed
			model = self.theta_to_model(theta)
			
			training_data_losses.append(self.loss(model, training_input, training_output, self.weight_decay_coefficient))
			validation_data_losses.append(self.loss(model, valid_input, valid_output, self.weight_decay_coefficient))
			if self.early_stop and (validation_data_losses[-1]< best_so_far['validation_loss']):
				best_so_far['theta'] =theta
				best_so_far['validation_loss'] = validation_data_losses[-1]
				best_so_far['after_n_iter'] = optimization_interation_i
		
		if self.early_stop:
			print('Training stopped after', best_so_far['after_n_iter'], 'with a validation loss of', best_so_far['validation_loss'])
			theta = best_so_far['theta']
		
		model = self.theta_to_model(theta)
		
		if self.n_iter!= 0:
			plt.plot(training_data_losses, c = 'b', label = 'training data')
			plt.plot(validation_data_losses, c = 'r', label = 'validation data')
			plt.legend()
			plt.show()
		
		print('Loss on the validation data', self.loss(model, valid_input, valid_output, self.weight_decay_coefficient))
		return model
	


if __name__ == '__main__':
	# data is stored on 'data.mat'
	data = sio.loadmat('data.mat')['data'][0][0]

	test_data = {'input': data[0][0][0][0].T, 'output':data[0][0][0][1].T}#training_data[0]: input: training_data[1]: output
	valid_data = {'input': data[1][0][0][1].T, 'output': data[1][0][0][0].T}
	training_data= {'input': data[2][0][0][0].T , 'output': data[2][0][0][1].T }
	numberRecognizer = NumberRecognizer()
	model = numberRecognizer.fit(training_data['input'], training_data['output'],
		valid_data['input'], valid_data['output'])
	print("Error rate: ", numberRecognizer.predict(model, test_data['input'], test_data['output']))
