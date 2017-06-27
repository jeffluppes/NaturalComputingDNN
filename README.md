# Natural Computing, team Eastern Screech Owl
Roel, Jeffrey and Jordi.

## A virtual machine
A virtual machine containing the entire repo in a working copy can be found at *a link will be placed here after deadline*

Using the virtual machine, you do not need to run the below instructions. 

## Getting started
To start, simply clone this repo using 
`git clone https://github.com/jeffluppes/NaturalComputingDNN` - Which works assuming git is installed. If not, it is also possible to download this entire repo the old-fashioned way by using the download button.  

From there on, things get a little bit complicated.

## Installing Anaconda
First off, install anaconda with Python 2.7 from https://www.continuum.io/downloads depending on your distribution. This should install most of the required software.

## Setting up the Gym environment
This project was made with OpenAI's gym ecosystem. We use Gym to load the game environments. While there are quite a lot of environments implemented, not all work under Windows 7/8/10, and there at least some that do not work even under Ubuntu.

Installing Gym can be done by

    git clone https://github.com/openai/gym
    cd gym
    pip install -e . # minimal install

or using `pip install Gym`


For the code in this repo, the choice between Ubuntu and Windows should not be relevant (this choice was mainly made for reproducability) but in case you, the reader, encounter errors, know that it can be oddly specific to your environment. If you are using Ubuntu we suggest Theano, as tensorflow is known to give errors.

## Installing Keras-rl

Keras-rl is a reinforcement learning library and can be installed with `pip install keras-rl`. Comes pre-packaged with Theano, and of course Keras. 

## Installing h5py
h5py is used to save and store model weights.   
`pip install h5py`

## This repo
The code used can be found in `../scripts/` and may be run by command-line commands, e.g. `python script.py`.

Reports and raw data can also be found in the designated folders. 

# A sample run
Assuming all of the above has been done, one can start any arbitrary script by typing
`python <script>.py`

![](http://puu.sh/wvThG/295ba87328.png)

The code will take a little while to produce anything, but eventually learning takes place with output being redirected to the terminal. 

![](http://puu.sh/wvTt9/1b5d88daad.jpg)

After completion, the terminal will show the results per episode. 

![](http://puu.sh/wvTE0/c774db4b68.png)

Like any other code, the output may be redirected or piped to a file (which is what we did in order to analyze the results, always a good tip!)