import numpy as np
from keras.applications import nasnet, inception_resnet_v2, inception_v3, resnet50
from keras.preprocessing import image
from keras import losses, metrics, callbacks, optimizers, activations, models, layers
from IPython.display import clear_output
from keras.models import Model
import visdom
//1
def _nasnet_large(num_classes, freezed=True):
    base_model = nasnet.NASNetLarge(weights='imagenet', include_top=False)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(num_classes, activation=activations.softmax, name='predictions')(x)
    if freezed:
        for layer in base_model.layers:
            layer.trainable = False
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model

def _inception_v3(num_classes, freezed=True):
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(num_classes, activation=activations.softmax, name='predictions')(x)
    if freezed:
        for layer in base_model.layers:
            layer.trainable = False
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model

def _inception_resnet_v2(num_classes, freezed=True):
    base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(num_classes, activation=activations.softmax, name='predictions')(x)
    if freezed:
        for layer in base_model.layers:
            layer.trainable = False
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model

def _resnet_50(num_classes, freezed=True):
    base_model = resnet50.ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(num_classes, activation=activations.softmax, name='predictions')(x)
    if freezed:
        for layer in base_model.layers:
            layer.trainable = False
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model

#image size para inception 299x299, para nasnet large 331x331 e para nasnet mobile/resnet50 224x224
img_size = 224
generator = image.ImageDataGenerator(rescale=1./255)
data = generator.flow_from_directory('/home/messias/Documentos/larvae/dataset', target_size=(img_size,img_size), shuffle=False)

# ESCOLHA O BASE MODEL DESCOMENTANDO

#base_model = _nasnet_large(data.num_classes, freezed=True)
#base_model = _inception_resnet_v2(data.num_classes, freezed=True)
base_model = _inception_v3(data.num_classes, freezed=True)
#base_model = _resnet_50(data.num_classes, freezed=True)

base_model.summary()
model = Model(input=base_model.input, output=base_model.get_layer('global_average_pooling2d_1').output)

output1 = model.predict_generator(data)
print(output1.shape)
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																												
output2 = data.classes
print(output2.shape)

aux = np.vstack(output2)
final_output = np.concatenate((output1, aux), axis=1)

# ESCOLHA O ARQUIVO DE SAIDA DESCOMENTANDO

#np.savetxt("nasnet_large.csv", final_output, delimiter=",")
#np.savetxt("inception_resnet_v2.csv", final_output, delimiter=",")
np.savetxt("inception_v3.csv", final_output, delimiter=",")
#np.savetxt("resnet_50.csv", final_output, delimiter=",")

