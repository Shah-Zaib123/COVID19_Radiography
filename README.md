# COVID19_Radiography

## Purpose
This is a project is for detecting COVID-19 using Radiographs using CNN.

## Dataset
You can download the training and testing dataset using following line of code.

```python
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
```

## Creating Model
Model being used is VGG19 and its all layers are kept trainable. The image size is **150x150**.

```python
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
  layer.trainable = True

layer1 = Flatten()(vgg.output)
layer2 = Dense(64, activation='relu')(layer1)
prediction = Dense(3, activation='softmax')(layer2)
# create a model object
model = Model(inputs=vgg.input, outputs=prediction)
```

Setting the optimizer and loss for the model.

```python
from keras.optimizers import SGD
sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
# view the structure of the model
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer= sgd,
  metrics=['accuracy']
)
```

## Data Augmentation

```python
train_test_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,
                                  validation_split = 0.20)

```
## Model Architecture

![alt text](https://github.com/MuhammadJunaidAkram/COVID19_Radiography/blob/main/images/arch.png?raw=true)

## Batch Size
The batch size is kept **64** for this model

## Training Model
Model is trained over **15 epochs**.
```python
r = model.fit_generator(generator=training_set,
                        validation_data=test_set,
                        epochs = 15,
                        steps_per_epoch = len(training_set),
                        validation_steps = len(test_set))
```

## Trained Model Weights
Weights of the trained model can be accessed using the following link.
```python
https://drive.google.com/file/d/19Rk8PTuE_Pfo-i9Y9pcHlFkGw3IYCxDj/view?usp=sharing
```

## Model Performance
**Accuracy**
The training accuracy of the model is 97.23% and testing accuracy of th model is 97.43%. Accuracy over each epoch can be observed in the following given graph.<br />
![alt text](https://github.com/MuhammadJunaidAkram/COVID19_Radiography/blob/main/images/accuracy.PNG?raw=true)

**Loss**
The training loss is 7.02% and validation loss is 7.84%. Loss over each epoch can be observed in the following given graph.<br />
![alt text](https://github.com/MuhammadJunaidAkram/COVID19_Radiography/blob/main/images/loss.PNG?raw=true)

**Confusion Matrix**
The confusion matrix of the model is as follows.<br />
![alt text](https://github.com/MuhammadJunaidAkram/COVID19_Radiography/blob/main/images/confusion_matrix.png?raw=true)

**Model Predictions**
Few model predictions are as follows.<br />
![alt text](https://github.com/MuhammadJunaidAkram/COVID19_Radiography/blob/main/images/pred_labels_results.png?raw=true)


