import tensorflow as tf
import math
import os
import random
import numpy as np
import scipy.io
import h5py
# import beamforming
with tf.device('/gpu:0'):
	def create_variable(shape, name):
		initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
		variable = tf.Variable(initializer(shape=shape), name = name)
		return variable

	def psnr_get(input_data, target_data):
		mse =np.sum(np.square(input_data - target_data))/np.size(input_data)
		max_value = target_data.max() - target_data.min()
		psnr_v = 20*np.log10(max_value)-10*np.log10(mse)
		return psnr_v

	weights = {
		'layer1_w' : create_variable([3,3,1,64], 'layer1_w'),
		'layer2_w' : create_variable([3,3,64,64], 'layer2_w'),
		'layer3_w' : create_variable([3,3,64,64], 'layer3_w'),
		'layer4_w' : create_variable([3,3,64,64], 'layer4_w'),

		'layer5_w' : create_variable([3,3,64,64], 'layer5_w'),
		'layer6_w' : create_variable([3,3,64,64], 'layer6_w'),
		'layer7_w' : create_variable([3,3,64,64], 'layer7_w'),
		'layer8_w' : create_variable([3,3,64,64], 'layer8_w'),

		'layer9_w' : create_variable([3,3,64,64], 'layer9_w'),
		'layer10_w' : create_variable([3,3,64,64], 'layer10_w'),
		'layer11_w' : create_variable([3,3,64,64], 'layer11_w'),
		'layer12_w' : create_variable([3,3,64,64], 'layer12_w'),

		'layer13_w' : create_variable([3,3,64,64], 'layer13_w'),
		'layer14_w' : create_variable([3,3,64,64], 'layer14_w'),
		'layer15_w' : create_variable([3,3,64,64], 'layer15_w'),
		'layer16_w' : create_variable([3,3,64,64], 'layer16_w'),

		'layer17_w' : create_variable([3,3,64,64], 'layer17_w'),
		'layer18_w' : create_variable([3,3,64,64], 'layer18_w'),
		'layer19_w' : create_variable([3,3,64,64], 'layer19_w'),
		'layer20_w' : create_variable([3,3,64,64], 'layer20_w'),

		'layer21_w' : create_variable([3,3,64,64], 'layer21_w'),
		'layer22_w' : create_variable([3,3,128,64], 'layer22_w'),
		'layer23_w' : create_variable([3,3,64,64], 'layer23_w'),
		'layer24_w' : create_variable([3,3,64,64], 'layer24_w'),

		'layer25_w' : create_variable([3,3,64,64], 'layer25_w'),
		'layer26_w' : create_variable([3,3,128,64], 'layer26_w'),
		'layer27_w' : create_variable([3,3,64,64], 'layer27_w'),
		'layer28_w' : create_variable([3,3,64,64], 'layer28_w'),

		'layer29_w' : create_variable([3,3,64,64], 'layer29_w'),
		'layer30_w' : create_variable([3,3,128,64], 'layer30_w'),
		'layer31_w' : create_variable([3,3,64,64], 'layer31_w'),
		'layer32_w' : create_variable([3,3,64,64], 'layer32_w'),
		'layer33_w' : create_variable([3,3,64,64], 'layer33_w'),

		'layer34_w' : create_variable([3,3,128,64], 'layer34_w'),
		'layer35_w' : create_variable([3,3,64,64], 'layer35_w'),
		'layer36_w' : create_variable([3,3,64,64], 'layer36_w'),
		'output_w' : create_variable([1,1,64,1], 'output_w')
		}

	def condata(data1, data2):
		next_input = tf.concat([data1,data2],3)
		return next_input

	def layer_model(input_data, hidden_w, strides = 1):
		conv_res  = tf.nn.conv2d(input_data, hidden_w, strides = [1, strides, strides, 1], padding = 'SAME')
		# batch_res = tf.contrib.layers.batch_norm(conv_res)
		batch_res, _, _ = tf.nn.fused_batch_norm(conv_res, scale = np.ones([64], dtype=np.float32) , offset = np.zeros([64], dtype=np.float32), epsilon=1e-3)
		relu_res = tf.nn.relu(batch_res)
		return  relu_res

	def layermodel_last(input_data, hidden_w, strides = 1):
		conv_res  = tf.nn.conv2d(input_data, hidden_w, strides = [1, strides, strides, 1], padding = 'SAME')
		return conv_res

	def model(image_data, weights):
		l1_res = layer_model(image_data, weights['layer1_w'])
		l2_res = layer_model(l1_res, weights['layer2_w'])
		l3_res = layer_model(l2_res, weights['layer3_w'])
		l4_res = layer_model(l3_res, weights['layer4_w'])

		l5_res = layer_model(l4_res, weights['layer5_w'])
		l6_res = layer_model(l5_res, weights['layer6_w']) 
		l7_res = layer_model(l6_res, weights['layer7_w'])
		l8_res = layer_model(l7_res, weights['layer8_w'])

		l9_res = layer_model(l8_res, weights['layer9_w']) 
		l10_res = layer_model(l9_res, weights['layer10_w'])  
		l11_res = layer_model(l10_res, weights['layer11_w'])
		l12_res = layer_model(l11_res, weights['layer12_w'])

		l13_res = layer_model(l12_res, weights['layer13_w'])
		l14_res = layer_model(l13_res, weights['layer14_w'])
		l15_res = layer_model(l14_res, weights['layer15_w'])
		l16_res = layer_model(l15_res, weights['layer16_w'])

		l17_res = layer_model(l16_res, weights['layer17_w'])
		l18_res = layer_model(l17_res, weights['layer18_w'])
		l19_res = layer_model(l18_res, weights['layer19_w'])
		l20_res = layer_model(l19_res, weights['layer20_w'])

		l21_res = layer_model(l20_res, weights['layer21_w'])
		l22_input = condata(l16_res, l21_res)
		l22_res = layer_model(l22_input, weights['layer22_w'])
		l23_res = layer_model(l22_res, weights['layer23_w'])
		l24_res = layer_model(l23_res, weights['layer24_w'])

		l25_res = layer_model(l24_res, weights['layer25_w'])
		l26_input = condata(l12_res, l25_res)
		l26_res = layer_model(l26_input, weights['layer26_w'])
		l27_res = layer_model(l26_res, weights['layer27_w'])
		l28_res = layer_model(l27_res, weights['layer28_w'])

		l29_res = layer_model(l28_res, weights['layer29_w'])
		l30_input = condata(l8_res, l29_res)
		l30_res = layer_model(l30_input, weights['layer30_w'])
		l31_res = layer_model(l30_res, weights['layer31_w'])
		l32_res = layer_model(l31_res, weights['layer32_w'])

		l33_res = layer_model(l32_res, weights['layer33_w'])
		l34_input = condata(l4_res, l33_res)
		l34_res = layer_model(l34_input, weights['layer34_w'])
		l35_res = layer_model(l34_res, weights['layer35_w'])
		l36_res = layer_model(l35_res, weights['layer36_w'])
		
		output = layermodel_last(l36_res, weights['output_w'])
		return output

	# Variables Setting
	# epoch = 400
	epoch = 600
	lamda = 10e-4
	wgt = 10e3
	# clip_value_min = -10e-2
	# clip_value_max = 10e-2
	learning_range = np.linspace(10e-7, 10e-9, num=epoch)

	# Placeholders Setting
	X = tf.placeholder(tf.float32, [None, 96, 64, 1])
	Y = tf.placeholder(tf.float32, [None, 96, 64, 1])
	learning_rate = tf.placeholder(tf.float32, shape=[])
	psnr_data = tf.placeholder(tf.float32, shape=[])
	traincost_holder = tf.placeholder(tf.float32, shape=[])
	validcost_holder = tf.placeholder(tf.float32, shape=[])

	# Model output
	imageoutput_res = model(X, weights)

	# l2 cost_Function
	regularizer = 0
	for i in range(np.shape(weights.values())[0]-1):
		weight_name = 'layer' + str(i+1) + '_w'
		weight = weights[weight_name]
		regularizer = regularizer + tf.nn.l2_loss(weight)

	trainloss = tf.sqrt(tf.reduce_sum(tf.pow((Y-imageoutput_res),2)))
	validloss = tf.sqrt(tf.reduce_sum(tf.pow((Y-imageoutput_res),2)))
	cost_train = tf.reduce_mean(trainloss + lamda * regularizer)
	cost_valid = tf.reduce_mean(validloss + lamda * regularizer)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_train)
	# # Gradient Clipping
	# opt_func = tf.train.GradientDescentOptimizer(learning_rate)
	# gvs = opt_func.compute_gradients(cost_train)
	# var = tf.trainable_variables()
	# capped_gvs = [(tf.clip_by_value(grad, clip_value_min, clip_value_max), var) for grad, var in gvs]
	# optimizer = opt_func.apply_gradients(capped_gvs)

	# Tensorboard Plot
	loss_train = tf.summary.scalar('loss_train', traincost_holder)
	merge_train = tf.summary.merge([loss_train])

	loss_valid = tf.summary.scalar('loss_valid', validcost_holder)
	psnr_save = tf.summary.scalar('psnr', psnr_data)
	merge_valid = tf.summary.merge([loss_valid, psnr_save])

	# Save and initialize
	saver = tf.train.Saver(weights)
	init = tf.global_variables_initializer()

	config = tf.ConfigProto(allow_soft_placement = True)
	with tf.Session(config=config) as sess:
		sess.run(init)

		writer = tf.summary.FileWriter('./summaries/', sess.graph)

		savefile_path = './Result/'
		traindata_path = './data/training/traindata8.mat'
		trainlabel_path = './data/training/trainlabel8.mat'
		validdata_path = './data/training/validdata8.mat'
		validlabel_path = './data/training/validlabel8.mat'
		f = h5py.File(traindata_path)
		f = f.get('traindata_selected_norm')
		traindata = np.array(f)
		f1 = h5py.File(trainlabel_path)
		f1 = f1.get('trainlabel_selected_norm')
		trainlabel = np.array(f1)
		f4 = h5py.File(validdata_path)
		f4 = f4.get('validdata_selected_norm')
		validdata = np.array(f4)
		f5 = h5py.File(validlabel_path)
		f5 = f5.get('validlabel_selected_norm')
		validlabel = np.array(f5)

		traindata = traindata *wgt
		validdata = validdata *wgt
		trainlabel = trainlabel * wgt
		validlabel = validlabel * wgt 

		
		train_depth =  25000
		valid_depth = 2500
		for cycle in range(epoch):
			print "train_step : " , cycle
			print "Training"
			# training
			costtrain_matrix = np.zeros((train_depth))

			for k in range(np.shape(traindata)[0]):
				train_planedata = traindata[k,:,:]
				traintarget_planedata = trainlabel[k,:,:]

				rand_num1 = random.random()
				if rand_num1 > 0.5:
					train_planedata = np.flip(train_planedata, axis = 0)
					traintarget_planedata = np.flip(traintarget_planedata, axis = 0)
				rand_num2 = random.random()
				if rand_num2 > 0.5:
					train_planedata = np.flip(train_planedata, axis = 1)
					traintarget_planedata = np.flip(traintarget_planedata, axis = 1)	

				train_planedata = np.reshape(train_planedata,[-1,96,64,1])
				traintarget_planedata = np.reshape(traintarget_planedata,[-1,96,64,1])
				
				_ , trainplane_loss = sess.run([optimizer, cost_train], feed_dict = {X : train_planedata , Y : traintarget_planedata, learning_rate : learning_range[cycle]})
				costtrain_matrix[k] = trainplane_loss / wgt
			train_loss = np.mean(costtrain_matrix)
			summar_train = sess.run(merge_train, feed_dict={traincost_holder : train_loss})
			print "train_loss : " , train_loss
			writer.add_summary(summar_train, cycle)
			
			saver.save(sess, './model/ultrasoundmodel.ckpt')		
			if (cycle+1) % 400 == 0:
				saver.save(sess, './model_400/ultrasoundmodel_epoch400.ckpt')
			if (cycle+1) % 500 == 0:
				saver.save(sess, './model_500/ultrasoundmodel_epoch500.ckpt')
			
			# Validation
			print "Validation"
			psnr_matrix = np.zeros((valid_depth))
			costvalid_matrix = np.zeros((valid_depth))
			for n in range(np.shape(validdata)[0]):
				valid_planedata = validdata[n,:,:]
				validtarget_planedata = validlabel[n,:,:]
				valid_planedata = np.reshape(valid_planedata, [-1,96,64,1])
				validtarget_planedata = np.reshape(validtarget_planedata, [-1,96,64,1])

				validrecon_planedata, validplane_loss = sess.run([imageoutput_res, cost_valid], feed_dict={X : valid_planedata, Y : validtarget_planedata})
				costvalid_matrix[n] = validplane_loss / wgt
				psnr_value = psnr_get(validrecon_planedata,validtarget_planedata)
				psnr_matrix[n] = psnr_value

			valid_loss = np.mean(costvalid_matrix)
			psnr_mean = np.mean(psnr_matrix)
			summar_valid = sess.run(merge_valid, feed_dict={psnr_data : psnr_mean, validcost_holder : valid_loss})
			print "valid_loss : " , valid_loss
			print "psnr : " , psnr_mean
			writer.add_summary(summar_valid, cycle)

			if cycle == 0:
	                		best_psnr = psnr_mean
			if best_psnr < psnr_mean:
	                		saver.save(sess, './model_best/ultrasoundmodelmodel.ckpt')
	                		best_psnr = psnr_mean






