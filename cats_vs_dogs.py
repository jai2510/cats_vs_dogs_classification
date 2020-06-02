train_dir = "............/cats_and_dogs_small/train"
validation_dir = ".........../cats_and_dogs_small/validation"
test_dir = ".........../cats_and_dogs_small/test"

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   vertical_flip=True,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    target_size=(150,150),
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                    batch_size=20,
                                                    target_size=(150,150),
                                                    class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir,
                                                    batch_size=20,
                                                    target_size=(150,150),
                                                    class_mode='binary')



from keras import models,layers,optimizers

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))

model = models.Sequential()

model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'],loss='binary_crossentropy')

model.summary()

history = model.fit_generator(train_generator,
                    epochs=5,
                    steps_per_epoch=100,
                    validation_data=validation_generator,
                    validation_steps=50)

test_loss,test_acc = model.evaluate_generator(test_generator,steps=50)

model.save("cats_vs_dogs.h5")
