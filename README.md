#Realtime Gesture recognition system based on deep learning (Deep learning) 

##Reference

````
Ji-Hae Kim, Gwang-Soo Hong and Byung-Gyu Kim, 
"Arm Gestures Recognition based on Deep Convolution and Recurrent Neural Network", 
Displays(Elsvier), 2018
````

### Anaconda with python 3.6 Download
````
https://www.continuum.io/downloads
````

### git clone
````
git clone https://github.com/GwangsHong/DeepGesture.git
````

### Anaconda virtual environment install
````
cd DeepGesture; conda env create
````
### virtual environment activate
````
activate deepGesture
````

### virtual environment deactivate
````
deactivate deepGesture
````

### virtual environment remove
````
conda remove --name deepGesture --all
````

###train example
````
python train.py /train/
````
###test example
````
python server.py
````