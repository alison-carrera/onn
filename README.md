# Online Neural Network (ONN)

This is a Pytorch implementation of the [Online Deep Learning: Learning Deep Neural Networks on the Fly](https://arxiv.org/abs/1711.03705) paper. This algorithm contains a new backpropagation approach called Hedge Backpropagation and it is useful for online learning. In this algorithm you model a overnetwork architeture and the algorithm will try to turn on or turn off some of the hidden layers automatically. This algorithm uses the first hidden layer to train/predict but if it is going bad it starts to use another layers automatically. For more informations read the paper in the 'References' section.

## Installing
```
pip install onn
```

## How to use
```python
#Importing Library
import numpy as np
from onn.OnlineNeuralNetwork import ONN

#Starting a neural network with feature size of 2, hidden layers expansible until 5, number of neuron per hidden layer = 10 #and two classes.
onn_network = ONN(features_size=2, max_num_hidden_layers=5, qtd_neuron_per_hidden_layer=10, n_classes=2)

#Do a partial training
onn_network.partial_fit(np.asarray([[0.1, 0.2]]), np.asarray([0]))
onn_network.partial_fit(np.asarray([[0.8, 0.5]]), np.asarray([1]))

#Predict classes
predictions = onn_network.predict(np.asarray([[0.1, 0.2], [0.8, 0.5]]))

Predictions -- array([1, 0])

```

## New features

- The algortihm works with batch now. (It is not recommended because this is an online approach. It is useful for experimentation.)
- The algorithm can use CUDA if available. (If the network is very small, it is not recommended. The CPU will process more fast.)

# Non-linear Contextual Bandit Algorithm (ONN_THS)

The ONN_THS acts like a non-linear contextual bandit (a reinforcement learning algorithm). This algorithm works with the non-linear exploitation factor (ONN) plus an exploration factor provided by Thompson Sampling algorithm. The ONN_THS works with 'select' and 'reward' actions. For more detailed examples, please look at the jupyter notebook file in this repository.

The great thing about this algorithm is that it can be used in an online manner and it has a non-linear exploitation. This algorithm can learn different kind of data in a reinforcement learning way.

## How to use
```python
#Importing Library
import numpy as np
from onn.OnlineNeuralNetwork import ONN_THS

#Starting a neural network with feature size of 2, hidden layers expansible until 5, number of neuron per hidden layer = 10 #and two classes.
onn_network = ONN_THS(features_size=2, max_num_hidden_layers=5, qtd_neuron_per_hidden_layer=10, n_classes=2)

#Select an action
arm_selected, exploration_factor = onn_network.predict(np.asarray([[0.1, 0.2]]))

#Reward an action
onn_network.partial_fit(np.asarray([[0.1, 0.2]]), np.asarray([arm_selected]), exploration_factor)

```

## Contributors
- [Alison de Andrade Carrera](https://github.com/alison-carrera)

## References
- [Online Deep Learning: Learning Deep Neural Networks on the Fly](https://arxiv.org/abs/1711.03705)
- [A Survey of Online Experiment Design with the Stochastic Multi-Armed Bandit](https://arxiv.org/pdf/1510.00757.pdf)
