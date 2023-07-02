import cv2
import numpy as np
import tensorflow as tf

class PrintedWrittenClassifier():
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
    
    def preprocess(self, image):
        IMG_SIZE = (256, 256)
        im = cv2.resize(image, IMG_SIZE, interpolation = cv2.INTER_AREA)
        im = np.expand_dims(im, axis = 0)
        return im
    
    def predict(self, img):
        return self.model.predict(img)[0, 0]
    
    def postprocess(self, y_pred):
        # printed
        if y_pred < 0.5:
            return 0
        # written    
        return 1

if __name__ == "__main__":
    model_path = "./model.h5"
    classifier = PrintedWrittenClassifier(model_path)
 
    image_path = "test_images/printed.tif"
    image = cv2.imread(image_path)
    image = classifier.preprocess(image)

    y_pred = classifier.predict(image)
    output = classifier.postprocess(y_pred)
    assert output == 0
