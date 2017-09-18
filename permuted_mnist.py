import tensorflow as tf
import numpy as np
from itertools import product
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/", one_hot=True)


#num_tasks_to_run = 2

num_epochs_per_task = 20


# Parameters for the intelligence synapses model.
param_c = 0.1
param_xi = 0.1


minibatch_size = 256
learning_rate = 0.001

def dendritic_clamp(omegas, N0, N1, N2, n_dendrites, num_tasks_to_run):

	W1c = np.zeros((N0, n_dendrites, N1), dtype = np.float32)
	W2c = np.zeros((N1, n_dendrites, N2), dtype = np.float32)
	b1c = np.zeros((1, n_dendrites, N1), dtype = np.float32)
	b2c = np.zeros((1, n_dendrites, N2), dtype = np.float32)

	omega_total_layer1 = np.zeros((N1, n_dendrites), dtype = np.float32)
	omega_total_layer2 = np.zeros((N2, n_dendrites), dtype = np.float32)

	if omegas == []:
		next_dendrite_layer1 = np.zeros((N1), dtype = np.int16)
		next_dendrite_layer2 = np.zeros((N2), dtype = np.int16)

	else:

		for n in range(num_tasks_to_run):

			omega_total_layer1 += np.mean(omegas[('W1',n)], axis=0).T + omegas[('b1',n)][0,:,:].T
			omega_total_layer2 += np.mean(omegas[('W2',n)], axis=0).T + omegas[('b2',n)][0,:,:].T

		next_dendrite_layer1 = np.argmin(omega_total_layer1, axis = 1)
		next_dendrite_layer2 = np.argmin(omega_total_layer2, axis = 1)

	for i in range(N1):
		W1c[:,next_dendrite_layer1[i], i] = 1
		b1c[0,next_dendrite_layer1[i], i] = 1

	for i in range(N2):
		W2c[:,next_dendrite_layer2[i], i] = 1
		b2c[0,next_dendrite_layer2[i], i] = 1

	#print('example dendrite clamp ', W1c[0,:, 0])

	return W1c, W2c, b1c, b2c

def weight_variable(input_size, n_dendrites, output_size, name):
	if n_dendrites > 0:
		return tf.Variable( tf.random_uniform([input_size,n_dendrites,output_size], -1.0/np.sqrt(input_size), 1.0/np.sqrt(input_size)), name = name)
	else:
		return tf.Variable( tf.random_uniform([input_size,output_size], -1.0/np.sqrt(input_size), 1.0/np.sqrt(input_size)) , name = name)

# Note: the main paper uses a larger network + dropout; both significantly improve the performance of the system.
N0 = 784
N1 = 400
N2 = 400
n_dendrites = 5
epsilon = 1e-4
n_tasks = 30

W1c = np.zeros((n_tasks, N0, n_dendrites, N1), dtype = np.float32)
W2c = np.zeros((n_tasks, N1, n_dendrites, N2), dtype = np.float32)
b1c = np.zeros((n_tasks, 1, n_dendrites, N1), dtype = np.float32)
b2c = np.zeros((n_tasks, 1, n_dendrites, N2), dtype = np.float32)

## Network definition -- a simple MLP with 2 hidden layers
x = tf.placeholder(tf.float32, shape=[None, N0])
y_tgt = tf.placeholder(tf.float32, shape=[None, 10])
task_vector = tf.placeholder(tf.float32, shape=[n_tasks])

W1_clamp = tf.placeholder(tf.float32, shape=[N0, n_dendrites, N1])
W2_clamp = tf.placeholder(tf.float32, shape=[N1, n_dendrites, N2])
b1_clamp = tf.placeholder(tf.float32, shape=[1, n_dendrites, N1])
b2_clamp = tf.placeholder(tf.float32, shape=[1, n_dendrites, N2])

W1 = tf.Variable(tf.random_uniform([N0,n_dendrites,N1], -1.0/np.sqrt(N0), 1.0/np.sqrt(N0)), name = 'W1')
b1 = tf.Variable(tf.zeros([1,n_dendrites,N1]), name = 'b1')

W2 = tf.Variable(tf.random_uniform([N1,n_dendrites,N2], -1.0/np.sqrt(N1), 1.0/np.sqrt(N1)), name = 'W2')
b2 = tf.Variable(tf.zeros([1,n_dendrites,N2]), name = 'b2')

Wo = tf.Variable(tf.random_uniform([N2,10], -1.0/np.sqrt(N2), 1.0/np.sqrt(N2)), name = 'Wo')
bo = tf.Variable(tf.zeros([1,10]), name = 'bo')

W1_effective = tf.reduce_sum(W1*W1_clamp, axis=1)
W2_effective = tf.reduce_sum(W2*W2_clamp, axis=1)
b1_effective = tf.reduce_sum(b1*b1_clamp, axis=1)
b2_effective = tf.reduce_sum(b2*b2_clamp, axis=1)

h1 = tf.nn.relu(tf.matmul(x,W1_effective) + b1_effective)
h2 = tf.nn.relu(tf.matmul(h1,W2_effective) + b2_effective)
y = tf.nn.softmax( tf.matmul(h2,Wo) + bo )


cross_entropy = -tf.reduce_sum( y_tgt*tf.log(y+epsilon) + (1.-y_tgt)*tf.log(1.-y+epsilon) )

optimizer = tf.train.AdamOptimizer(learning_rate)
#optimizerAdam = tf.train.AdamOptimizer(learning_rate)

## Implementation of the intelligent synapses model
variables = [W1, b1, W2, b2, Wo, bo]

small_omega_var = {}
previous_weights_mu_minus_1 = {}
big_omega_var = {}
aux_loss = 0.0

reset_small_omega_ops = []
update_small_omega_ops = []
update_big_omega_ops = []
#for var, task_num in zip(variables, range(n_tasks)):
for var, task_num in product(variables, range(n_tasks)):
	small_omega_var[var.op.name, task_num] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
	previous_weights_mu_minus_1[var.op.name, task_num] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
	big_omega_var[var.op.name, task_num] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

	aux_loss += tf.reduce_sum(tf.multiply( big_omega_var[var.op.name, task_num], tf.square(previous_weights_mu_minus_1[var.op.name, task_num] - var) ))

	reset_small_omega_ops.append( tf.assign( previous_weights_mu_minus_1[var.op.name, task_num], var ) )
	reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name, task_num], small_omega_var[var.op.name, task_num]*0.0 ) )

	update_big_omega_ops.append( tf.assign_add( big_omega_var[var.op.name, task_num],  task_vector[task_num]*tf.div(small_omega_var[var.op.name, task_num], \
		(param_xi + tf.square(var-previous_weights_mu_minus_1[var.op.name, task_num])))))

# After each task is complete, call update_big_omega and reset_small_omega
update_big_omega = tf.group(*update_big_omega_ops)
#new_big_omega_var = big_omega_var

# Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
reset_small_omega = tf.group(*reset_small_omega_ops)


# Gradient of the loss function for the current task
gradients = optimizer.compute_gradients(cross_entropy, var_list=variables)

# Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
gradients_with_aux = optimizer.compute_gradients(cross_entropy + param_c*aux_loss, var_list=variables)


"""
Apply any applicable weights masks to the gradient and clip
"""
capped_gvs = []
for grad, var in gradients_with_aux:
	capped_gvs.append((tf.clip_by_norm(grad, 5), var))

# This is called every batch
#print(small_omega_var.keys())
for i, (grad,var) in enumerate(gradients_with_aux):
	for j in range(n_tasks):
		update_small_omega_ops.append( tf.assign_add( small_omega_var[var.op.name, j], task_vector[j]*learning_rate*capped_gvs[i][0]*gradients[i][0] ) ) # small_omega -= delta_weight(t)*gradient(t)

update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!




train = optimizer.apply_gradients(capped_gvs)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_tgt,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

## Permuted MNIST

# Generate the tasks specifications as a list of random permutations of the input pixels.
task_permutation = []
for task in range(n_tasks):
	task_permutation.append( np.random.permutation(N0) )


avg_performance = []
first_performance = []
last_performance = []
omegas = []

for task in range(n_tasks):
	print("Training task: ",task+1,"/",n_tasks)

	# one-hot vector indicating current task
	tv = np.zeros((n_tasks), dtype = np.float32)
	tv[task] = 1

	W1c[task,:,:,:], W2c[task,:,:,:], b1c[task,:,:,:], b2c[task,:,:,:] = dendritic_clamp(omegas, N0, N1, N2, n_dendrites, n_tasks)
	#print('dend c ', W1c[task,0,:,0])

	#sess.run(optimizerAdam)
	t0 = time.time()
	for epoch in range(num_epochs_per_task):
		if epoch%1==0:
			print("\t Epoch ",epoch,' Epoch time ', time.time() - t0)
			t0 = time.time()

		for i in range(mnist.train.num_examples//minibatch_size):
		#for i in range(2):
			# Permute batch elements
			batch = mnist.train.next_batch(minibatch_size)
			batch = ( batch[0][:, task_permutation[task]], batch[1] )

			sess.run([train, update_small_omega, big_omega_var], feed_dict={x:batch[0], y_tgt:batch[1], task_vector: tv, \
				W1_clamp: W1c[task,:,:,:], W2_clamp: W2c[task,:,:,:], b1_clamp: b1c[task,:,:,:], b2_clamp: b2c[task,:,:,:]})

	sess.run(update_big_omega, feed_dict = {task_vector: tv})
	omegas = sess.run(big_omega_var)

	sess.run( reset_small_omega )

	# Print test set accuracy to each task encountered so far
	avg_accuracy = 0.0
	for test_task in range(task+1):
		test_images = mnist.test.images

		# Permute batch elements
		test_images = test_images[:, task_permutation[test_task]]

		acc = sess.run(accuracy, feed_dict={x:test_images, y_tgt:mnist.test.labels, \
			W1_clamp: W1c[test_task,:,:,:], W2_clamp: W2c[test_task,:,:,:], b1_clamp: b1c[test_task,:,:,:], b2_clamp: b2c[test_task,:,:,:]}) * 100.0
		avg_accuracy += acc

		if test_task == 0:
			first_performance.append(acc)
		if test_task == task:
			last_performance.append(acc)

		print("Task: ",test_task," \tAccuracy: ",acc)

	avg_accuracy = avg_accuracy/(task+1)
	print("Avg Perf: ",avg_accuracy)

	avg_performance.append( avg_accuracy )
	print()
	print()




import matplotlib.pyplot as plt
tasks = range(1,n_tasks+1)
plt.plot(tasks, first_performance)
plt.plot(tasks, last_performance)
plt.plot(tasks, avg_performance)
plt.legend(["Task 0 (t=i)", "Task i (t=i)", "Avg Task (t=i)"], loc='lower right')
plt.xlabel("Task")
plt.ylabel("Accuracy (%)")
plt.ylim([50, 100])
plt.xticks(tasks)
plt.show()
