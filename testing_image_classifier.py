from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import model_from_json
import pandas as pd
import json
import cv2
import numpy as np

test_data_dir = 'data/test'

with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('first_try.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1)

predicted_class_indices=np.where(pred>0.5,'dogs','cats')

predictions = predicted_class_indices.tolist()

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)