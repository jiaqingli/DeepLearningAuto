#!/usr/bin/env python
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed, Reshape
from keras.layers import LSTM, Conv2D, MaxPooling2D, Activation
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import sequence
import time


def _load_data():

	data=np.load('autorally_data.npz', encoding = 'bytes')
	#500 datapoints in total
	img = data['observations'][()][b'img_left']  #64*128*3
	actions = data['actions']
	return img, actions



def main():
#-------------------------------------Model---------------------------------------------------#
	model = Sequential()
	
	# define CNN model
	model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3),activation='relu'), input_shape=(None, 64, 128, 3)))
	model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3),activation='relu')))	
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

	model.add(TimeDistributed(Conv2D(64, kernel_size=(3, 3),activation='relu')))
	model.add(TimeDistributed(Conv2D(64, kernel_size=(3, 3),activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	
	model.add(TimeDistributed(Conv2D(128, kernel_size=(3, 3),activation='relu')))
	model.add(TimeDistributed(Conv2D(128, kernel_size=(3, 3),activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	model.add(TimeDistributed(Flatten()))
	
	# define LSTM model
	model.add(LSTM(128, return_sequences = True))
	model.add(LSTM(64, return_sequences = True))
	model.add(LSTM(32))
	
	model.add(Dense(32))
	model.add(Dense(2))
	#model.add(Activation('tanh'))
	

	# compile model
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
	model.summary()

#-----------------------------------Data-----------------------------------------------#
	imgs, actions = _load_data()
	
	imgs_input = []
	actions_output = []

	training_set = 100
	past_img_num = 6
	predict_step = 10
	first_img_num = past_img_num * predict_step 
	#in this case: 20, 15, 10, 5, 0 -> 25

	for i in range(training_set):
		imgs_input_delayed = []
		actions_input_delayed = []
		for j in range(past_img_num):
			img = imgs[i + j*predict_step]
			img_x = img_to_array(img)
			imgs_input_delayed.append(img_x)

		imgs_input.append(imgs_input_delayed)
		actions_output.append(actions[i+first_img_num])

	print("imgs_input shape")
	print(np.array(imgs_input).shape[0])
	print(np.array(imgs_input).shape[1])
	print(np.array(imgs_input).shape[2])
	print(np.array(imgs_input).shape[3])
	print(np.array(imgs_input).shape[4])
	print("actions_output shape")
	print(np.array(actions_output).shape[0])
	print(np.array(actions_output).shape[1])

	evaluation_set = 1
	imgs_test_input = []
	actions_test_output = []
	#start_num = training_set + first_img_num #125
	start_num = 200

	for i in range(evaluation_set):
		print('test images indexes are:')		
		imgs_input_test_delayed = []
		for j in range(past_img_num):
			print(start_num+i+(j*predict_step))
			img = imgs[start_num+i+(j*predict_step)]
			img_x = img_to_array(img)
			imgs_input_test_delayed.append(img_x)
		imgs_test_input.append(imgs_input_test_delayed)
		print('test action index is', start_num+i+first_img_num)
		actions_test_output.append(actions[start_num+i+first_img_num])
	
#---------------------------------------train-------------------------------------------#
	start_time = time.time()
	model.fit(np.array(imgs_input), np.array(actions_output), batch_size=32, epochs=2)
	end_time = time.time()
	print('training time: ', end_time - start_time)
#---------------------------------------evaluate----------------------------------------#
	#evaluate
	stime = time.time()
	test_loss, test_accuracy = model.evaluate(np.array(imgs_test_input), np.array(actions_test_output), verbose=True)
	etime = time.time()
	print('evaluation time: ', etime-stime)
	print('test_accuracy: ', test_accuracy)
	print('test_loss: ', test_loss)

#-------------------------------------prediction-----------------------------------------#
	pre_s_time = time.time()	
	prediction = model.predict(np.array(imgs_test_input))
	pre_e_time = time.time()	
	print('prediction time is ', pre_e_time - pre_s_time)
	print('prediction result is ', prediction)
	print('real output is ', actions_test_output)

	
main()
