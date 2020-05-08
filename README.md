# Detecting-Equipment-Failure-Using-LSTM-Autoencoder

This is a rough Autoencoder for detecting an equipment life based on its repair time for every notification flag on some malfunction. 

We specifically chose **LSTM Autoencoder** based on their ability to remember information for later use in the network, making them well suited for temporal data analysis that evolves over time. Also, LSTM cells have the ability to include multivariate features in analysis. LSTM networks are a sub-type of recurrent neural networks (RNN). 

The autoencoder architecture basically creates a compressed representation of the primary driving features of that data i.e. **"identity function"** and then learns to reconstruct it again. For example, input an image of a cat, autoencoder will compress that data to the core constituents that make up the cat picture and then learn to recreate the original picture from the compressed data.

>The rationale behind using this architecture for anomaly detection is to detect the **reconstruction error** generated from training the model on "normal data". This is done so that when the model encounters data that is new and attempts to reconstruct it, we will see an increase in the reconstruction error as the model was never trained to accurately recreate items from the new data.

We used **TensorFlow** as our backend and **Keras** as the model development library.
<br> Optimizer in use : **Adam** </br>

Since our data has a frequency of failure distribution, we need not use **Fourier transform**. In case the data is in time domain, you need to apply Fourier transform to change it into frequency domain.

Point to note: **LSTM cells expect a 3 dimensional tensor of the form [data samples, time steps, features]**. 

