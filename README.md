#VQA Demo

This is a simple Demo of Visual Question answering which uses pretrained models (see models/CNN and models/VQA) to answer a given question about the given image.

## Dependency 

(Python dependencies and versions used listed on requirements.txt)

1. Keras version 1.0.2
   * Modular deep learning library based on python

2. Theano/Tensorflow 
    * As backend for Keras, you may choose either one. 
    * For the development of this project, I used Theano 0.8.0

3. scikit-learn
   * Quintessential machine library for python

4. Spacy version 0.100.6
    * Used to load Glove vectors (word2vec)
    * You may have to upgrade your Spacy to use Glove vectors (default is Goldberg Word2Vec)
    * To install Glove Vectors
      * python -m spacy.en.download
      * sputnik --name spacy install en_glove_cc_300_1m_vectors

5. OpenCV 
    * OpenCV is used only to resize the image and change the color channels,
    * You may use other libraries as long as you can pass a 224x224 BGR Image (NOTE: BGR and not RGB)
  
6. VGG 16 Pretrained Weights
    * Please download the weights file [vgg16_weights.h5](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view)

##Usage

> python demo.py -image_file_name `path_to_file` -question "Question to be asked"

e.g 

> python demo.py -image_file_name test.jpg -question "Is there a man in the picture?"


if you have prefer to use Theano backend and if you have GPU you may want to run like this

> THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1,mode=FAST_RUN' python demo.py -image_file_name test.jpg -question "What vechile is in the picture?"


Expected Output :
095.2 %  train
00.67 %  subway
00.54 %  mcdonald's
00.38 %  bus
00.33 %  train station


##Runtime

  * GPU (Titan X) Theano optimizer=fast_run       : 51.3 seconds
  * GPU (Titan X) Theano optimizer=fast_compile   : 47.5 seconds
  * CPU (i7-5820K CPU @ 3.30GHz                   : 35.9 seconds (Is this strange or not ?)

##iPython Notebook

Jupyter/iPython Notebook has been provided with more examples and interactive tutorial.
<https://github.com/iamaaditya/VQA_Demo/blob/master/Visual_Question_Answering_Demo_in_python_notebook.ipynb>

NOTE:
See the comments on demo.py for more information on the model and methods

