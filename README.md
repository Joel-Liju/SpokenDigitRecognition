This needs to be filled in before submission
so if you wanted you can make a virtual environment, which is what I did. To create that, you do

`python3 -m venv .venv`

after that you can activate the environment and run the virtual environment.

However, the only necessary step for this part is to run the command.

> pip install -r requirements.txt

which will install all the required files, and any test data that you use, please make sure you put it in the 
* testdata

folder. As we don't want to populate the github with data.

---wav2spec---
This program converts all .wav files within a provided path into spectrogram .png images.
The images are formated (no whitespace, axis) to be provided to image AI. 
To run wav2spec.py:
python wav2spec.py folderContainingWavs

Few of the libraies we are using include:-
* [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/)
* [Tensorflow](https://www.tensorflow.org/api_docs/python/tf)
* [scipy](https://docs.scipy.org/doc/scipy/)
* [matplotlib](https://matplotlib.org/)
* [tqdm](https://tqdm.github.io/)

you can look at the [requirements](./requirements.txt) for the accurate packages required for this project.

Another great tool we found is called Data Version Control, which is basically git for data. As we used data for training our models, we needed the data to be accessed from our computers and if we changed any data, we needed us to be able to access them. 
You can read more about that [here](https://dvc.org/doc)

Finally the last tool we used and require users to have so that they can run this project is, [tkinter](https://docs.python.org/3/library/tkinter.html) . This is a really useful tool for people to get started with GUI programming in python.

# Project Description.
## Aim
We were aiming to make a software, that gives the option of selecting an audio, or recording yourself, and then run an AI model on it, to identify which number it is from 0-9.

## How to run it.
In order to run it, you have to ensure that you have 
> python 3.11

plus all the requirements installed. After that, you need to download the model that was used for this project specifically, which can be done using dvc.

Where you need to pull the dvc model from our drive. However, if this is not possible either make your own model using the [imageClassification.py](./imageClassification.py) script, or shoot us a message.

## How does it work.
It works using the idea of image recognition, where we trained the model using spectrograms for different numbers from 0 - 9 and then when we analyze audio, we convert that into spectrogram and feed it through the classifier and then present the result.

## how to use the software.

So there are a few ways which you can add audio to test, first you can select a .wav file from your compute which is about 1 second long. Or you can record your own voice then run it through the model. 
So make sure you update the sample rate appropriately, as if not things might not work as expected. We mostly use 44100 Hz as a default, but feel free to test that out as well.
