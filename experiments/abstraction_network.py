# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import sys, numpy

class abstraction_network:
	def __init__(self,sess,params,num_abstract_states):
		self.sess=sess
		self.obs_size=params['obs_size']
		self.num_abstract_states=num_abstract_states
		self.learning_rate=params['learning_rate_for_abstraction_learning']

		with tf.variable_scope('abstraction_scope'):
			self.obs=tf.placeholder(tf.float32, [None, self.obs_size], name = 'obs')
			self.Pr_a_given_z=tf.placeholder(tf.float32, [None,self.num_abstract_states], name = 'prob_of_all_a_given_z')
			h=self.obs
			for _ in range(params['abstraction_network_hidden_layers']):
				h=tf.layers.dense(
		            inputs = h,
		            units = params['abstraction_network_hidden_nodes'],
		            activation = tf.nn.relu)

			self.logits=tf.layers.dense(
	            inputs = h,
	            units = self.num_abstract_states)

			self.Pr_z_given_s = tf.nn.softmax(self.logits)
			self.Pr_z_given_s=tf.clip_by_value(
				    self.Pr_z_given_s,
				    clip_value_min=0.0001,
				    clip_value_max=0.9999,
				)

			self.loss=-tf.reduce_mean(tf.multiply(self.Pr_a_given_z,tf.log(self.Pr_z_given_s)))
														  
											
									
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

			#self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
	def predict(self,samples):

		# print("predict function")
		# print(f"samples predict: {samples} type: {type(samples)}")
		# samples_np = numpy.array(samples, dtype=object)
		# print("Shape of samples:", samples_np.shape)

		# # Print the shape of self.obs
		# print("Shape of self.obs:", self.obs.shape)

		# # Print the expected shape of self.obs
		# print("Expected shape for self.obs:", self.obs.get_shape().as_list())

		# Ensure that the shapes match before running the session
		# assert samples_np.shape[1] == self.obs.get_shape().as_list()[1], f"Dimension mismatch: samples_np.shape[1]={samples_np.shape[1]}, self.obs.shape[1]={self.obs.get_shape().as_list()[1]}"

		# feed_dict = {self.obs: samples_np}
		# print(f"feed dict {feed_dict}")
		
		# li=self.sess.run(self.Pr_z_given_s,feed_dict=feed_dict)
	
		li=self.sess.run(self.Pr_z_given_s,feed_dict={self.obs:samples})
		return li

	def train(self,samples,a_in_z):
		s_li=[]
		prob_a_in_z_for_mdp_li=[]

		for sample in samples:

			s_li.append(numpy.array(sample[0]))
			action=sample[1]
			mdp_number=sample[2]
			a_in_z_for_mdp=a_in_z[:,mdp_number]
			prob_a_in_z_for_mdp=[float(action==x) for x in a_in_z_for_mdp]
			prob_a_in_z_for_mdp_li.append(numpy.array(prob_a_in_z_for_mdp))
		_,l=self.sess.run([self.optimizer,self.loss],feed_dict={self.obs:s_li,
									   self.Pr_a_given_z:prob_a_in_z_for_mdp_li})
		return l

