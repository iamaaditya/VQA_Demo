#VQA Demo

This is a simple Demo of Visual Question answering which uses pretrained models (see models/CNN and models/VQA) to answer a given question about the given image.

## Dependency

1. Keras
   * Modular deep learning library based on python

2. Theano/Tensorflow 
    * As backend for Keras, you may choose either one. 
    * For the development of this project, I used Theano 0.8.0

3. scikit-learn
   * Quintessential machine library for python

4. Spacy 
    * Used to load Glove vectors (word2vec)
    * You may have to upgrade your Spacy to use Glove vectors (default is Goldberg Word2Vec)
    * To upgrade & install Glove Vectors
      * pip install --upgrade spacy
      * sputnik —name spacy install en
      * sputnik --name spacy install en_glove_cc_300_1m_vectors

5. OpenCV 
    * OpenCV is used only to resize the image and change the color channels,
    * You may use other libraries as long as you can pass a 224x224 BGR Image (NOTE: BGR and not RGB)

##Usage

> python demo.py --file `path_to_file` --question "Question to be asked"

e.g 

> python demo.py --file test.jpg --question "Is there a man in the picture?"


if you have prefer to use Theano backend and if you have GPU you may want to run like this

> THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1,mode=FAST_RUN' python demo.py -image_file_name test.jpg -question "What vechile is in the picture?"


Expected Output :

78.32 %  train
01.11 %  truck
00.98 %  passenger
00.95 %  fire truck
00.68 %  bus


##Runtime

  * GPU (Titan X) Theano optimizer=fast_run       : 117 seconds
  * GPU (Titan X) Theano optimizer=fast_compile   : 122 seconds
  * CPU (i7-5820K CPU @ 3.30GHz                   : 160 seconds


NOTE:
See the comments on demo.py for more information on the model and methods

