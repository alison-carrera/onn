# Online Neural Network (ONN)

This is a Pytorch implementation of the [Online Deep Learning: Learning Deep Neural Networks on the Fly](https://arxiv.org/abs/1711.03705) paper. This algorithm contains a new backpropagation approach called Hedge Backpropagation and it is useful for online learning. In this algorithm you model a overnetwork architeture and the algorithm will try to turn on or turn off some of the hidden layers automatically. This algorithm uses the first hidden layer to train/predict but if it is going bad it starts to use another layers automatically. For more informations read the paper in the 'References' section.

## Installing
```
pip install onn
```

## How to use
```python

```

## New features

- The algortihm works with batch now. (It is not recommended because this is an online approach. It is useful for experimentation.)
- The algorithm can use CUDA if available. (If the network is very small, it is not recommended. The CPU will process more fast.)

## Contributors
- [Alison de Andrade Carrera](https://github.com/alison-carrera)
- [Fábio Silva Vilas Boas](https://github.com/fabiosvb)

## References
- [Online Deep Learning: Learning Deep Neural Networks on the Fly](https://arxiv.org/abs/1711.03705)
