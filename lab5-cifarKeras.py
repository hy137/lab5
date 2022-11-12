# -*- coding:utf-8 -*-
import tensorflow as tf
import time
from loaddata import load_data

IMGdim = 32			#图像大小
IMGchn = 3			#图像通道
CLSsum = 10			#分类类别数量


learnRate = 1e-3	# learning rate
ep = 2			# epochs
bs = 250 		# batch size
keep_prob = 0.75	# drop-out 概率
model_save_path = './save_model/'


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

	# 卷积层（非线性变化）
    basemodel = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding='same',activation='relu',input_shape=[IMGdim, IMGdim, IMGchn]),
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),  #池化,output 16x16

        tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2), #池化,output 8x8
        ])

	# 直连层（线性分类器）
    fcmodel = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512),
        tf.keras.layers.Dropout(keep_prob),
        tf.keras.layers.Dense(units=CLSsum, activation='softmax')
        ])
		
	# 整个网络模型
    model = tf.keras.Sequential()
    model.add(basemodel)
    model.add(fcmodel)

    model.build(input_shape=[None, IMGdim, IMGdim, IMGchn])
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate,beta_1=0.9,beta_2=0.999,epsilon=1e-8,decay=0.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

	# 训练
    model.fit(x=x_train, y=y_train, batch_size=bs,epochs=ep,validation_split=0.1,verbose=1)


	# 测试
    loss,accuracy = model.evaluate(x_test,y_test)
	
    print('\ntest loss=',loss)
    print('accuracy=',accuracy)

    savetime=time.time()
    model.save_weights(model_save_path+str(int(savetime))+'model.h5', save_format='h5')
    basemodel.save_weights(model_save_path+str(int(savetime))+'basemodel.h5', save_format='h5')


