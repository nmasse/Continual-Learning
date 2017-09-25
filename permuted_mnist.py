import tensorflow as tf
import numpy as np
import get_cifar as gc
from itertools import product
import time, os

##################
### Parameters ###
##################

task_type = 'mnist'

n_tasks = 30
num_epochs_per_task = 2

param_c		= 0.1
param_xi	= 0.1

minibatch_size	= 256
learning_rate	= 0.001

if task_type == 'mnist':
	N0 = 784
	No = 10
	#num_examples = mnist.train.num_examples
elif task_type == 'cifar':
	N0 = 3*1024
	No = 10

N1 = 400
N2 = 400

n_dendrites = 5
ce_epsilon = 1e-4


#############################
### Startup administrivia ###
#############################

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Reset TensorFlow before running anything
tf.reset_default_graph()

# Import MNIST data
if task_type == 'mnist':
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("/tmp/", one_hot=True)

	num_examples = mnist.train.num_examples
	test_img_func = mnist.test.images

	next_batch = mnist.train.next_batch
elif task_type == 'cifar':
	cifar = gc.data_set('cifar-10')

	num_examples = cifar.num_examples
	test_img_func = cifar.test_images
	next_batch = cifar.next_batch
	

##################################
### Model function definitions ###
##################################

def dendritic_clamp(omegas):

	# Initialize clamps
	W1c = np.zeros((N0, n_dendrites, N1), dtype=np.float32)
	W2c = np.zeros((N1, n_dendrites, N2), dtype=np.float32)
	b1c = np.zeros((1, n_dendrites, N1), dtype=np.float32)
	b2c = np.zeros((1, n_dendrites, N2), dtype=np.float32)

	# Initialize omega totals
	omega_total_layer1 = np.zeros((N1, n_dendrites), dtype=np.float32)
	omega_total_layer2 = np.zeros((N2, n_dendrites), dtype=np.float32)

	if omegas == []:
		next_dendrite_layer1 = np.zeros((N1), dtype=np.int16)
		next_dendrite_layer2 = np.zeros((N2), dtype=np.int16)
	else:
		for n in range(n_tasks):
			omega_total_layer1 += np.mean(omegas[('W1',n)], axis=0).T + omegas[('b1',n)][0,:,:].T
			omega_total_layer2 += np.mean(omegas[('W2',n)], axis=0).T + omegas[('b2',n)][0,:,:].T
		next_dendrite_layer1 = np.argmin(omega_total_layer1, axis=1)
		next_dendrite_layer2 = np.argmin(omega_total_layer2, axis=1)

	for i in range(N1):
		W1c[:,next_dendrite_layer1[i],i] = 1
		b1c[0,next_dendrite_layer1[i],i] = 1

	for i in range(N2):
		W2c[:,next_dendrite_layer2[i],i] = 1
		b2c[0,next_dendrite_layer2[i],i] = 1

	#print('example dendrite clamp ', W1c[0,:, 0])
	return W1c, W2c, b1c, b2c


def weight_variable(input_size, output_size, name, no_dend=False):
	if n_dendrites==0 or no_dend:
		shape = [input_size,output_size]
	else:
		shape = [input_size,n_dendrites,output_size]

	bound = 1.0/np.sqrt(input_size)
	return tf.Variable(tf.random_uniform(shape, -bound, bound), name=name)


def bias_variable(size, name, no_dend=False):
	if n_dendrites==0 or no_dend:
		shape = [1,size]
	else:
		shape = [1,n_dendrites,size]

	return tf.Variable(tf.zeros(shape), name=name)


########################
### Model definition ###
########################

print('\nSetting up model.')

# Note: the main paper uses a larger network + dropout; both significantly improve the performance of the system.
W1c = np.zeros((n_tasks, N0, n_dendrites, N1), dtype=np.float32)
W2c = np.zeros((n_tasks, N1, n_dendrites, N2), dtype=np.float32)
b1c = np.zeros((n_tasks, 1, n_dendrites, N1), dtype=np.float32)
b2c = np.zeros((n_tasks, 1, n_dendrites, N2), dtype=np.float32)

# Network definition -- a simple MLP with 2 hidden layers
x = tf.placeholder(tf.float32, shape=[None, N0])
y_tgt = tf.placeholder(tf.float32, shape=[None, 10])
task_vector = tf.placeholder(tf.float32, shape=[n_tasks])

# Defining clamp placeholders
W1_clamp = tf.placeholder(tf.float32, shape=[N0, n_dendrites, N1])
W2_clamp = tf.placeholder(tf.float32, shape=[N1, n_dendrites, N2])
b1_clamp = tf.placeholder(tf.float32, shape=[1, n_dendrites, N1])
b2_clamp = tf.placeholder(tf.float32, shape=[1, n_dendrites, N2])

# Defining parameter matrices
W1 = weight_variable(N0, N1, 'W1')
b1 = bias_variable(N1, 'b1')

W2 = weight_variable(N1, N2, 'W2')
b2 = bias_variable(N2, 'b2')

Wo = weight_variable(N2, No, 'Wo', no_dend=True)
bo = bias_variable(No, 'bo', no_dend=True)

# Defining clamp interactions
W1_effective = tf.reduce_sum(W1*W1_clamp, axis=1)
W2_effective = tf.reduce_sum(W2*W2_clamp, axis=1)
b1_effective = tf.reduce_sum(b1*b1_clamp, axis=1)
b2_effective = tf.reduce_sum(b2*b2_clamp, axis=1)

# Defining network behavior
h1 = tf.nn.relu(tf.matmul(x,W1_effective) + b1_effective)
h2 = tf.nn.relu(tf.matmul(h1,W2_effective) + b2_effective)
y = tf.nn.softmax(tf.matmul(h2,Wo) + bo)

# Loss calculation and optimizer setup
cross_entropy = -tf.reduce_sum( y_tgt*tf.log(y+ce_epsilon) + (1.-y_tgt)*tf.log(1.-y+ce_epsilon) )
optimizer = tf.train.AdamOptimizer(learning_rate)
#optimizerAdam = tf.train.AdamOptimizer(learning_rate)


########################################################
### Implementation of the intelligent synapses model ###
########################################################

variables = [W1, b1, W2, b2, Wo, bo]

small_omega_var = {}
previous_weights_mu_minus_1 = {}
big_omega_var = {}
aux_loss = 0.0

reset_small_omega_ops = []
update_small_omega_ops = []
update_big_omega_ops = []
for var, task_num in product(variables, range(n_tasks)):
	small_omega_var[var.op.name, task_num] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
	previous_weights_mu_minus_1[var.op.name, task_num] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
	big_omega_var[var.op.name, task_num] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

	aux_loss += tf.reduce_sum(tf.multiply(big_omega_var[var.op.name, task_num], tf.square(previous_weights_mu_minus_1[var.op.name, task_num] - var)))

	reset_small_omega_ops.append(tf.assign(previous_weights_mu_minus_1[var.op.name, task_num], var))
	reset_small_omega_ops.append(tf.assign(small_omega_var[var.op.name, task_num], small_omega_var[var.op.name, task_num]*0.0))

	update_big_omega_ops.append(tf.assign_add( big_omega_var[var.op.name, task_num], task_vector[task_num]*tf.div(small_omega_var[var.op.name, task_num], \
		(param_xi + tf.square(var-previous_weights_mu_minus_1[var.op.name, task_num])))))

# After each task is complete, call update_big_omega and reset_small_omega
update_big_omega = tf.group(*update_big_omega_ops)
#new_big_omega_var = big_omega_var

# Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
reset_small_omega = tf.group(*reset_small_omega_ops)


##################################################################
### Compute loss and gradients, then optimize and get accuracy ###
##################################################################

# Gradient of the loss function for the current task
gradients = optimizer.compute_gradients(cross_entropy, var_list=variables)

# Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
gradients_with_aux = optimizer.compute_gradients(cross_entropy + param_c*aux_loss, var_list=variables)

# Apply any applicable weights masks to the gradient and clip
capped_gvs = []
for grad, var in gradients_with_aux:
	capped_gvs.append((tf.clip_by_norm(grad, 5), var))

# Generate w_k update items based on gradients
# This is called every batch
#print(small_omega_var.keys())
for (i, (grad,var)), j in product(enumerate(gradients_with_aux), range(n_tasks)):
	to_update	= task_vector[j]*learning_rate*capped_gvs[i][0]*gradients[i][0]
	assignation = tf.assign_add(small_omega_var[var.op.name,j], to_update)
	update_small_omega_ops.append(assignation) # small_omega -= delta_weight(t)*gradient(t)

update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!

# Define training object
train = optimizer.apply_gradients(capped_gvs)

# Get accuracy
prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_tgt,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

print('Model setup complete.\n')


################################
### Run Permuted MNIST tasks ###
################################

# Generate the tasks specifications as a list of random permutations of the input pixels.
task_permutation = []
for task in range(n_tasks):
	task_permutation.append(np.random.permutation(N0))

avg_performance = []
first_performance = []
last_performance = []
omegas = []

# Begin training loop
for task in range(n_tasks):
	print("Training task: ",task+1,"/",n_tasks)

	# One-hot vector indicating current task
	tv = np.zeros((n_tasks), dtype=np.float32)
	tv[task] = 1

	# Get the dendritic clamp for this task
	W1c[task,:,:,:], W2c[task,:,:,:], b1c[task,:,:,:], b2c[task,:,:,:] = dendritic_clamp(omegas)
	#print('dend c ', W1c[task,0,:,0])

	#sess.run(optimizerAdam)
	t0 = time.time()
	for epoch in range(num_epochs_per_task):
		if epoch%1==0:
			print("\t Epoch ",epoch,' Epoch time ', time.time() - t0)
			t0 = time.time()

		for i in range(num_examples//minibatch_size):
		#for i in range(2):
			# Permute batch elements
			# batch = (input, output)
			t1 = time.time()
			batch = next_batch(minibatch_size)
			print(time.time()-t1)
			batch = (batch[0][:,task_permutation[task]], batch[1])

			sess.run([train, update_small_omega, big_omega_var], feed_dict={x:batch[0], y_tgt:batch[1], task_vector: tv, \
				W1_clamp: W1c[task,:,:,:], W2_clamp: W2c[task,:,:,:], b1_clamp: b1c[task,:,:,:], b2_clamp: b2c[task,:,:,:]})

	sess.run(update_big_omega, feed_dict = {task_vector: tv})
	omegas = sess.run(big_omega_var)
	sess.run(reset_small_omega)

	# Print test set accuracy to each task encountered so far
	avg_accuracy = 0.0
	for test_task in range(task+1):
		test_images = test_img_func

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
	print("Avg Perf: ",avg_accuracy,"\n\n\n")

	avg_performance.append(avg_accuracy)




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
