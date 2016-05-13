import os, argparse
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
    
# File paths for the model, all of these except the CNN Weights are 
# provided in the repo, See the models/CNN/README.md to download VGG weights
VQA_model_file_name     = 'models/VQA/VQA_MODEL.json'
VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'models/CNN/vgg16_weights.h5'

# Chagne the value of verbose to 0 to avoid printing the progress statements
verbose = 1

def get_image_model(CNN_weights_file_name):
    ''' Takes the CNN weights file, and returns the VGG model update 
    with the weights. Requires the file VGG.py inside models/CNN '''
    from models.CNN.VGG import VGG_16
    image_model = VGG_16(CNN_weights_file_name)

    # this is standard VGG 16 without the last two layers
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # one may experiment with "adam" optimizer, but the loss function for
    # this kind of task is pretty standard
    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return image_model

def get_image_features(image_file_name, CNN_weights_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the 
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 1000))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    im = im.transpose((2,0,1)) # convert the image to RGBA

    
    # this axis dimension is required becuase VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0) 

    image_features[0,:] = get_image_model(CNN_weights_file_name).predict(im)[0]
    return image_features

def get_VQA_model(VQA_model_file_name, VQA_weights_file_name):
    ''' Given the VQA model and its weights, compiles and returns the model '''

    # thanks the keras function for loading a model from JSON, this becomes
    # very easy to understand and work. Alternative would be to load model
    # from binary like cPickle but then model would be obfuscated to users
    vqa_model = model_from_json(open(VQA_model_file_name).read())
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

def get_question_features(question):
    ''' For a given question, a unicode string, returns the timeseris vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, len(tokens), 300))
    for j in xrange(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


def main():
    ''' accepts command line arguments for image file and the question and 
    builds the image model (VGG) and the VQA model (LSTM and MLP) 
    prints the top 5 response along with the probability of each '''

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

    # This task here is represented as a classification into a 1000 top answers
    # this means some of the answers were not part of trainng and thus would 
    # not show up in the result.
    # These 1000 answers are stored in the sklearn Encoder class
    labelencoder = joblib.load(label_encoder_file_name)
    for label in reversed(y_sort_index[0,-5:]):
        print str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label)

if __name__ == "__main__":
    main()
