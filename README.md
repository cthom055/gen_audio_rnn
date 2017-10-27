# Generative Audio

## Introduction and Overview

This is the repository for the networks that generate audio.

* [main.ipynb](main.ipynb): *This is the go to file that contains the model, the audio dataset loader and the audio generation. Naturally the IPython notebook makes it easy to explore the data.*
* [main_large_dataset.ipynb](main_large_dataset.ipynb): *This example takes on multiple samples to train for much longer. Use this once you have [main.ipynb](main.ipynb) working and experiment with the parameters to achieve different results*

* [old_main.py](old_main.py): *This is a older file that had the initial vinilla TensorFlow code in it. It creates the model via the methods from model.py. Higher level TensorFlow wrappers such as tflearn and slim tended to deliver better results so we migrated to using those in the main.ipynb. This demonstrates a lot of the usefulness of the AudioDatasetGenerator in the running of a tf session via the get_next_batch and is_new_epoch functions.*
* [old_model.py](old_model.py): *This contained the vinilla TensorFlow code to replicate Sam's Keras model. As mentioned earlier higher level wrappers provided better results to we moved to those.*
* [old_phase_gen.py](old_phase_gen.py): *This is the original method of phase reconstruction. Currently we now use griffin lim, which can be found in the audio_dataset_generator.py under the method with the same name. There is also a phase reconstruction network in progress in the phase_reconstruction.ipynb file, although this as of the time of writing is achieving poor results.*
* [assets](assets): *This is the folder that contains the audio for training. The current code in main.ipynb instanciates a AudioDatasetGenerator object that will pass the path to the assets folder so it can create the necessary sequences of fft magnitudes.*

-----

## Running on B0rk

To access the b0rk via ssh:
```
ssh username@igor.gold.ac.uk
ssh eavi@158.223.52.45
```

Check the docker processes:
```
sudo docker ps
```
If the container is running copy the container id and run the ipython notebook:
```
sudo docker exec -it <container id> /run_jupyter.sh
```

Else open a new container with the saved docker image, forward ipython notebook port to 8882 then run the notebook:
```
sudo nvidia-docker run -it -p 8882:8888 golb0rk/lstmsynth /run_jupyter.sh
```

#### Forwarding ports to access notebook

You can use any port numbers, if it complains try a different port - I find it easiest to keep a tunnel open (with -f) from my igor into b0rk. Then all I have to do is tunnel from my local machine into igor.

ssh tunnel from Igor to b0rk (whilst sshed into igor):
```
ssh -N -f -L localhost:8883:localhost:8882 eavi@158.223.52.45
```
then tunnel from your local machine to igor
```
ssh -N -L localhost:8884:localhost:8883 username@igor.gold.ac.uk
```
now ```localhost:8884``` should work in your browser!

#### Saving the docker container state

Detach from the docker with ```Ctrl+p```+```Ctrl+q``` (exit or ctrl+e will halt the container).

Then from outside the docker (eavi):
```
sudo docker commit <container id> golb0rk/lstmsynth
```

-------

## Running locally

To run this locally, it is assumed that Python 3.5, TensorFlow 1.1++ and the latest version of TFLearn are all installed on your system.

### Instructions

Generally the model found in the main.ipynb file is the best place to start hacking around. Have at it!

### Upload audio files to Jupyter

You can upload your own training data through the iPython file browser.

Locate the audio data path set in the main script. In this example it's been set to: **assets/test_samples**
```
audio_data_path      = "assets/test_samples"
```

[![upload.png](https://s20.postimg.org/h1szvmhy5/upload.png)](https://postimg.org/image/y2bw4auzd/)

Upload a .wav file

[![Upload_2.png](https://s20.postimg.org/flhhdhf19/Upload_2.png)](https://postimg.org/image/azld54ti1/)

Confirm upload

[![force_new.png](https://s20.postimg.org/gdk5coj8d/force_new.png)](https://postimg.org/image/rpwqugrx5/)

When using a new set of samples, make sure **force_new_dataset** is set to true

[![Restart.png](https://s20.postimg.org/7l36ozy3h/Restart.png)](https://postimg.org/image/n6ki8ya1l/)

When running a new test, you'll need to restart the kernal in the main notebook and re-run all the scripts. Tip: in the Jupyter toolbar, go to **_Cell_** -> **_Run All_**

### Code

There are a few things to note when building networks.

Audio data can be managed using the AudioDatasetGenerator. Care should be taken to remove any existing .npy files from whatever folder the target wavs are stored as the class will attempt to load those first. To force a new dataset, just pass true in the load method where force_new_dataset is.

```python
dataset = AudioDatasetGenerator(fft_size, window_size, hop_size, sequence_length, sample_rate)
dataset.load(audio_data_path, force_new_dataset)
```

The above code is fairly trivial - one other thing that might be unclear is the sequence_length parameter. This simply means how many n frames of fft magnitudes there should be for every prediction of a frame of fft magnitudes by the model.

____

### To do:
* Implement fast griffin lim / implement good phase reconstruction network.
* Get working instructions on how to get ipython notebook of b0rk running locally.
* Modify current instructions to allow multiple users to modify same image - currently all ip's are used to ipython forwarding.
* Perhaps try something like seq2seq for the generation of frames.
* Add variable layer mlp / highway layers at the end for easier experimentation.
* Experiment with deconvolutions at the end?
