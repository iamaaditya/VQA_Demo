import os, argparse
import cv2, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
from models.CNN.VGG import VGG_16

import spacy
    
VQA_model_file_name     = 'models/VQA/VQA_MODEL.json'
VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'models/CNN/vgg16_weights.h5'
image_file_name         = 'test.jpg'
question                = unicode("Where is the man?", "utf-8")

def get_image_model(CNN_weights_file_name):
    image_model = VGG_16(CNN_weights_file_name)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return image_model

def get_image_features(image_file_name, CNN_weights_file_name):
    image_features = np.zeros((1, 4096))
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    im = im.transpose((2,0,1)) # convert the image to RGBA
    im = np.expand_dims(im, axis=0)
    image_features[0,:] = get_image_model(CNN_weights_file_name).predict(im)[0]
    return image_features

def get_VQA_model(VQA_model_file_name, VQA_weights_file_name):
    vqa_model = model_from_json(open(VQA_model_file_name).read())
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

def get_question_features(question):
    word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    timesteps = len(word_embeddings(question)) #question sorted in descending order of length
    nb_samples = 1
    word_vec_dim = 300
    question_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
    tokens = word_embeddings(question)
    for j in xrange(len(tokens)):
        if j<timesteps:
                question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


def main():
    vqa_model = get_VQA_model(VQA_model_file_name, VQA_weights_file_name)
    image_features = get_image_features(image_file_name, CNN_weights_file_name)
    question_features = get_question_features(question)

    X = [question_features, image_features]

    y_output = vqa_model.predict(X)
    y_sort_index = np.argsort(y_output)

    labelencoder = joblib.load(label_encoder_file_name)
    for label in reversed(y_sort_index[0,-5:]):
        print str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label)

if __name__ == "__main__":
    main()
