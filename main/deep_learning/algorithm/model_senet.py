import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop
import main.deep_learning.config as config
from main.deep_learning.data_preprocessing.utils import NameMapID
import matplotlib.pyplot as plt
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class Densenet:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name_map = NameMapID()
        self.num_samples = config.TRAIN_IMAGE_NUM
        self.test_samples = config.TEST_IMAGE_NUM
        self.train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        self.test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
        )

    def fit(self, lr=0.01, batch_size=32, epoch=1000):
        densenet = DenseNet121(include_top=False, weights='imagenet', input_tensor=None, input_shape=self.input_shape,
                               pooling=None, classes=self.output_shape[0])
        x = densenet.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.output_shape[0], activation='softmax')(x)
        model_densenet = Model(inputs=densenet.input, outputs=predictions)
        print(model_densenet.summary())

        model_densenet.compile(optimizer=RMSprop(lr=lr),
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

        train_data = self.train_datagen.flow_from_directory(os.path.join(config.DATA_PATH, config.TRAIN_PATH),
                                                            target_size=(self.input_shape[0], self.input_shape[1]),
                                                            classes=self.name_map.classes(),
                                                            class_mode='categorical',
                                                            batch_size=batch_size,
                                                            shuffle=True)
        test_data = self.test_datagen.flow_from_directory(os.path.join(config.DATA_PATH, config.TEST_PATH),
                                                          target_size=(self.input_shape[0], self.input_shape[1]),
                                                          classes=self.name_map.classes(),
                                                          class_mode='categorical',
                                                          batch_size=batch_size,
                                                          shuffle=True)

        output_model_file = '/home/hehaoming/checkpoint/checkpoint-{epoch:02d}e-val_acc_{val_acc:.2f}.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(output_model_file, monitor='val_acc', verbose=1,
                                                     save_best_only=True)
        history = model_densenet.fit_generator(train_data,
                                               steps_per_epoch=math.ceil(self.num_samples / batch_size),
                                               epochs=epoch,
                                               callbacks=[checkpoint],
                                               validation_data=test_data,
                                               validation_steps=math.ceil(self.test_samples / batch_size))

        self.plot_training(history)

    @staticmethod
    def plot_training(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'r-')
        plt.plot(epochs, val_acc, 'b')
        plt.title('Training and validation accuracy')
        plt.figure()
        plt.plot(epochs, loss, 'r-')
        plt.plot(epochs, val_loss, 'b-')
        plt.title('Training and validation loss')
        plt.show()


if __name__ == "__main__":
    model = Densenet((256, 256, 3), (45,))
    model.fit()
