**COSC 4P98**

Alex Freer, 6452551

Joel Jacob, 6603245


# Spoken Digit Recognition


## **December 30, 2022**


# Overview

We created a Python based application to use an Artificial Neural Net (ANN) to analyze digital audio to determine spoken digits (0-9). We created and trained this model using a dataset consisting of 33,000 audio recordings gathered from multiple sources (Jakobovski; Soerenab). The audio recordings were converted to spectrograms and the data was split into their respective classifications (0-9). The ANN was then built to identify the classification of a given spectrogram, as the underlying idea behind this voice recognition problem is image classification. 


# Training Data

As previously mentioned, we gathered spoken digit audio samples from several sources, each consisting of almost 1 second of audio, where the beginning and trailing silence have been kept to a minimum. We then created a utility `wav2spec.py ` which converts all .wav files in a provided directory into spectrograms of a standardized size (227x227 px, as per our chosen architecture). This was approximately half the original size. After that all the spectrograms were saved into an output directory as .png files, we then created a utility `datasplitter.py` which split the data into separate folders (i.e. classifications) to finally be used in the training of the ANN. 


# Building The Model

We used the TensorFlow Keras python library to create a convolutional neural net (CNN) based on the AlexNet architecture. We chose this architecture because it was designed for an image of size 227x227 and to be trained on a large dataset. The sequential model consists of the following layers;


<table>
  <tr>
   <td>Layer
   </td>
   <td># Filters
   </td>
   <td>Kernel size
   </td>
   <td>Stride
   </td>
   <td>Padding
   </td>
   <td>Layer Size
   </td>
   <td>Activation Function
   </td>
  </tr>
  <tr>
   <td>Input (Rescaling)
   </td>
   <td>-
   </td>
   <td>-
   </td>
   <td>-
   </td>
   <td>-
   </td>
   <td>227x227x3
   </td>
   <td>-
   </td>
  </tr>
  <tr>
   <td>Convolution
   </td>
   <td>96
   </td>
   <td>11x11
   </td>
   <td>4
   </td>
   <td>-
   </td>
   <td>55x55x96
   </td>
   <td>ReLU
   </td>
  </tr>
  <tr>
   <td>MaxPooling
   </td>
   <td>-
   </td>
   <td>3x3
   </td>
   <td>2
   </td>
   <td>-
   </td>
   <td>27x27x96
   </td>
   <td>-
   </td>
  </tr>
  <tr>
   <td>Convolution
   </td>
   <td>256
   </td>
   <td>5x5
   </td>
   <td>1
   </td>
   <td>2
   </td>
   <td>27x27x256
   </td>
   <td>ReLU
   </td>
  </tr>
  <tr>
   <td>MaxPooling
   </td>
   <td>-
   </td>
   <td>3x3
   </td>
   <td>2
   </td>
   <td>-
   </td>
   <td>13x13x256
   </td>
   <td>-
   </td>
  </tr>
  <tr>
   <td>Convolution
   </td>
   <td>384
   </td>
   <td>3x3
   </td>
   <td>1
   </td>
   <td>1
   </td>
   <td>13x13x384
   </td>
   <td>ReLU
   </td>
  </tr>
  <tr>
   <td>Convolution
   </td>
   <td>384
   </td>
   <td>3x3
   </td>
   <td>1
   </td>
   <td>1
   </td>
   <td>13x13x384
   </td>
   <td>ReLU
   </td>
  </tr>
  <tr>
   <td>Convolution
   </td>
   <td>256
   </td>
   <td>3x3
   </td>
   <td>1
   </td>
   <td>1
   </td>
   <td>13x13x256
   </td>
   <td>ReLU
   </td>
  </tr>
  <tr>
   <td>MaxPooling
   </td>
   <td>-
   </td>
   <td>3x3
   </td>
   <td>2
   </td>
   <td>-
   </td>
   <td>6x6x256
   </td>
   <td>-
   </td>
  </tr>
  <tr>
   <td colspan="7" >Flatten
   </td>
  </tr>
  <tr>
   <td>Dense
   </td>
   <td>-
   </td>
   <td>-
   </td>
   <td>-
   </td>
   <td>-
   </td>
   <td>4096
   </td>
   <td>ReLU
   </td>
  </tr>
  <tr>
   <td colspan="7" >Dropout (0.5)
   </td>
  </tr>
  <tr>
   <td>Dense
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
   <td>4096
   </td>
   <td>ReLU
   </td>
  </tr>
  <tr>
   <td colspan="7" >Dropout (0.5)
   </td>
  </tr>
  <tr>
   <td>Output (Dense)
   </td>
   <td>-
   </td>
   <td>-
   </td>
   <td>-
   </td>
   <td>-
   </td>
   <td>10
   </td>
   <td>softmax
   </td>
  </tr>
</table>


This produced a total of 58,322,314 trainable parameters. After training was completed the model was saved into the directory `model` and uploaded using Data Version Control (DVC) to our data repository. 


# Data Version Control (DVC)

DVC is built to make ML models shareable and reproducible. It is designed to handle large files, data sets, machine learning models, and metrics as well as code.

For this project, DVC was used to share, and track the data used to train the Neural Network and also the model itself was also shared. You can read more about it using the DVC docs (DVC docs)

In order to be able to get the data, you have to have access to the drive which contains it. After that, you can use the command 

`dvc pull -r myremote`

As this command uses the .dvc files and the .dvc folder to pull the data down for usage. 


# Speech Recognition Pipeline 

The pipeline to analyze digital audio using the model goes as follows,



1. Get Audio Input  

    The user chooses a .wav file or records themself speaking.

2. Spectrogram 

    From either source, the audio data is converted into a spectrogram and saved in the directory `testdata` and loaded back as a PIL Image.  

3. Load Model

    The model is restored from the saved directory `model`. 

4. Prediction

   The spectrogram is fed through the restored model to get the most likely spoken digit and confidence of the prediction. 


# GUI




![Our GUI](/assets/images/gui.jpg "gui")




1. Select Audio
2. Get AI Prediction
3. Start Recording 
4. Stop Recording 
5. Play Recorded
6. Clear Recorded Data
7. Update Sample Rate
8. Current Sample Rate
9. New Sample Rate Input Field
10. Spectrogram of audio 
11. AI prediction 


# How to run the Project

Things to ensure are installed before running this Project are:-



* scipy
* numpy
* PyAudio
* Pillow
* Tkinter
* Tensorflow
* Python version 3.11

The best way to set this up would be to first install Python 3.11, then make a virtual environment 

`python3 -m venv path/to/folder`

Then once the environment is activated, you can just use `pip install -r requirement.txt` to pip install all the dependencies. 

After that, you just need to run the main.py file, 

`python main.py`

However, one thing to ensure is that you have a folder called ‘model’ which contains the details about the model within it. This is a template folder structure created by Tensorflow.

When the GUI is up and running, then you can select an already existing audio in your computer, or record yourself saying a number from 0-9, just make sure that the audio has little to no silence in the beginning and the end of the audio clip, and that it is less than a second.


# Experimentation 

Through experimentation, it was found the model provided better predictions when using a sampling rate of 8kHz. This was likely due to the training data being recorded in this sampling rate. In addition, we also found that using the magnitude mode provided by the library is better for classification than the default mode. As it provided more clearer differentiation between the numbers. Shorter recordings with less beginning and trailing silence also provided better results. In conclusion, the accuracy of predictions will be affected with the pronunciation, length and overall quality of the provided recording. 


# Future Works

In the future, we want to expand this software to be able to recognize multiple digits for a sequence, i.e. if we were to say 1 2 3, separately, we want to be able to pick out on these numbers. In addition to that, we want to expand into more numbers than just 0-9 and be able to use this AI to understand speech as well.


# Known Issues



* If you leave space in the front while recording, the system is not able to identify which number it is.
* The same happens if we leave a lot of space in the back as well.
* You can still feed it garbage and it does spit out garbage as well. As in, it cannot differentiate between number and other words yet.
* Changing the sample rate from 8kHz to another number, doesn’t help with improving the accuracy. In addition to that, because it was trained on data from 8kHz the ML model is not correctly able to identify the audio.


# Workload split



* Joel
    * Gathering the data and making the spectrograms
    * Trained the Neural network and setting up DVC for that
    * Helped improve the Machine Learning Model.
* Alex
    * Creating and updating the GUI
    * Developing the pipeline to be able to run the model on the input audio using the GUI
    * Improved the Machine learning model, to improve overall accuracy. 


# Works Cited


    Becker, Sören, et al. “Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals.” _CoRR_, vol. abs/1807.03418, 2018.


    “Data Version Control.” _DVC_, https://dvc.org/doc. Accessed 30 December 2022.


    Jakobovski. _free-spoken-digit-dataset_. 12 August 2020. _GitHub_, https://github.com/Jakobovski/free-spoken-digit-dataset.


    Python. _tkinter — Python interface to Tcl/Tk_. https://docs.python.org/3/library/tkinter.html.


    Saxena, Shipra. “Alexnet Architecture | Introduction to Architecture of Alexnet.” _Analytics Vidhya_, 19 March 2021, https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/. Accessed 30 December 2022.


    scipy. _API Reference_. Open a WAV file. https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html.


    scipy. _API Reference_. Write a NumPy array as a WAV file. https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html.


    Soerenab. _AudioMNIST_. 2018. _GitHub_, https://github.com/soerenab/AudioMNIST.


    TensorFlow. _Image classification_. https://www.tensorflow.org/tutorials/images/classification#a_basic_keras_model.
