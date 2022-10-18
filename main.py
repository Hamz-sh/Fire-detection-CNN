import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from matplotlib.image import imread


# Storing the path of the data
data_dir = 'C:/Users/Alsho/Desktop/university/finished/grad. project-2/data'
print(os.listdir(data_dir))

# loading model after training it on google colab
model = load_model('my_model_n2.h5') 
model.summary()

# Specifying the required shape
image_shape = (668, 922, 3)
test_path = data_dir + '/new data'


# Showing an image
imge = imread(test_path + '/0/istockphoto-471395415-612x612.jpg')
print(imge)
plt.imshow(imge)
plt.show()


# Random image transformations (images should be separated in different classified files for the function to work)
image_gen = ImageDataGenerator(fill_mode='nearest',
                               rescale=1/255)
test_gen = image_gen.flow_from_directory(test_path,
                                         target_size=image_shape[:2], color_mode='rgb',
                                         batch_size=32, class_mode='binary', shuffle=False)
print(test_gen)

# Predictions of the model over new data
pred = model.predict(test_gen)

print(pred)
print(test_gen.classes)

