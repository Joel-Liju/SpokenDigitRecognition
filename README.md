This needs to be filled in before submission
so if you wanted you can make a virtual environment, which is what I did. To create that, you do

`python3 -m venv .venv`

after that you can activate the environment and run the virtual environment.

However, the only necessary step for this part is to run the command.

> pip install -r requirements.txt

which will install all the required files, and any test data that you use, please make sure you put it in the 
* test data

folder. As we don't want to populate the github with data.

---wav2spec---
This program converts all .wav files within a provided path into spectrogram .png images.
The images are formated (no whitespace, axis) to be provided to image AI. 
To run wav2spec.py:
python wav2spec.py folderContainingWavs