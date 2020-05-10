from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import MobileNetV2
import tensorflow as tf
import json
import os
import PIL.Image as im
import numpy as np

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path
train_dir = image_path + "train"
validation_dir = image_path + "val"

im_height = 224
im_width = 224
batch_size = 16
epochs = 20


# 该预处理函数是根据tensorflow官方提供的预训练参数的预处理方法，在slim文件夹中有preprocessing文件夹，该文件夹中有preprocessing_factory.py文件，打开该文件便知道mobilenet v2模型的预处理是用inception_preprocessing方法，他在preprocessing文件夹中的inception_preprocessing.py文件中的preprocess_fortrain函数，该函数返回的tensor是-1到1之间，具体处理的代码行在函数的最后两行
def pre_function(img):
    # img = im.open('test.jpg')
    # img = np.array(img).astype(np.float32)
    # 对图像的预处理没有像resnet那样将每个通道减去通道的均值，而仅仅是将像素值缩放到-1到1之间
    img = img / 255.
    img = (img - 0.5) * 2.0
    return img


# data generator with data augmentation
train_image_generator = ImageDataGenerator(horizontal_flip=True,
                                           preprocessing_function=pre_function)

validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')
total_train = train_data_gen.n

# get class dict
class_indices = train_data_gen.class_indices

# transform value and key of dict
inverse_dict = dict((val, key) for key, val in class_indices.items())
# write dict into json file
json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')
# img, _ = next(train_data_gen)
total_val = val_data_gen.n

model = MobileNetV2(num_classes=5)
# feature.build((None, 224, 224, 3))  # when using subclass model
# 载入的参数是在公开数据集上预训练的参数，而且需要通过处理，将官方提供的预训练参数转换成我们需要的
model.load_weights('pretrain_weights.ckpt')
for layer_t in model.layers[:-1]:
    layer_t.trainable = False
model.summary()
# 模型一共有200万参数

# using keras low level api for training
# 因为网络最后没有使用softmax，那么在损失函数中设置from_logits为True
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = loss_object(labels, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, output)


@tf.function
def test_step(images, labels):
    output = model(images, training=False)
    t_loss = loss_object(labels, output)

    test_loss(t_loss)
    test_accuracy(labels, output)


best_test_loss = float('inf')
for epoch in range(1, epochs + 1):
    train_loss.reset_states()  # clear history info
    train_accuracy.reset_states()  # clear history info
    test_loss.reset_states()  # clear history info
    test_accuracy.reset_states()  # clear history info

    # train
    for step in range(total_train // batch_size):
        images, labels = next(train_data_gen)
        train_step(images, labels)

        # print train process
        rate = (step + 1) / (total_train // batch_size)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        acc = train_accuracy.result().numpy()
        print("\r[{}]train acc: {:^3.0f}%[{}->{}]{:.4f}".format(epoch, int(rate * 100), a, b, acc), end="")
    print()

    # validate
    for step in range(total_val // batch_size):
        test_images, test_labels = next(val_data_gen)
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    if test_loss.result() < best_test_loss:
        best_test_loss = test_loss.result()
        model.save_weights("./save_weights/resMobileNetV2.ckpt", save_format="tf")
