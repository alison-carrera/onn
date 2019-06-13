# Online Neural Network (ONN)

This is a Pytorch implementation of the [Online Deep Learning: Learning Deep Neural Networks on the Fly](https://arxiv.org/abs/1711.03705) paper. This algorithm contains a new backpropagation approach called Hedge Backpropagation and it is useful for online learning. In this algorithm you model a overnetwork architeture and the algorithm will try to turn on or turn off some of the hidden layers automatically. This algorithm uses the first hidden layer to train/predict but if it is going bad it starts to use another layers automatically. For more informations read the paper in the 'References' section.

## Installing
```
pip install onn
```

## How to use
```python
#Importing Library
from onn.OnlineNeuralNetwork import ONN

#Starting a neural network with feature size of 2, hidden layers expansible until 5, number of neuron per hidden layer = 10 #and two classes.
onn_network = ONN(features_size=2, max_num_hidden_layers=5, qtd_neuron_per_hidden_layer=10, n_classes=2)

#Do a partial training
onn_network.partial_fit(np.asarray([[0.1, 0.2]]), np.asarray([0]))
onn_network.partial_fit(np.asarray([[0.8, 0.5]]), np.asarray([1]))

#Predict classes
predictions = onn_network.predict(np.asarray([[0.1, 0.2], [0.8, 0.5]]))

Predictions -- array([1, 1])

#Predict classes probabilities
predictions = onn_network.predict_proba(np.asarray([[0.1, 0.2], [0.8, 0.5]]))

Predictions -- array([[0.5048331 , 0.50083154],[0.49516693, 0.49916846]], dtype=float32)
```

## New features

- The algortihm works with batch now. (It is not recommended because this is an online approach. It is useful for experimentation.)
- The algorithm can use CUDA if available. (If the network is very small, it is not recommended. The CPU will process more fast.)

## Contributors
- [Alison de Andrade Carrera](https://github.com/alison-carrera)
- [Fábio Silva Vilas Boas](https://github.com/fabiosvb)

## References
- [Online Deep Learning: Learning Deep Neural Networks on the Fly](https://arxiv.org/abs/1711.03705)
