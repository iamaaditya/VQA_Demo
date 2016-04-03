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

# Chagne the value of verbose to 0 to avoid printing the progress statements
verbose = 1

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
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, len(tokens), 300))
    for j in xrange(len(tokens)):
        # if j<timesteps:
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_file_name', type=str, default='test.jpg')
    parser.add_argument('-question', type=str, default='What vechile is in the picture?')
    args = parser.parse_args()

    
    if verbose : print("\n\n\nLoading image features ...")
    image_features = get_image_features(args.image_file_name, CNN_weights_file_name)

    if verbose : print("Loading question features ...")
    question_features = get_question_features(unicode(args.question, 'utf-8'))

    if verbose : print("Loading VQA Model ...")
    vqa_model = get_VQA_model(VQA_model_file_name, VQA_weights_file_name)


    if verbose : print("\n\n\nPredicting result ...") 
    y_output = vqa_model.predict([question_features, image_features])
    y_sort_index = np.argsort(y_output)

    labelencoder = joblib.load(label_encoder_file_name)
    for label in reversed(y_sort_index[0,-5:]):
        print str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label)

if __name__ == "__main__":
    main()
