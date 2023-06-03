## Introduction
An academic project about a Recommender System realized for the Intelligent Web course. 
This Recsys is based on the _Neighborhood-based Neural Collaborative Filtering_ model and 
the Deep Learning theory. 
The dataset used for the experimental analysis is [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/).

## Requirements

- [Python 3.10](https://www.python.org/downloads/release/python-3100/)
- [Numpy 1.24.2](https://pypi.org/project/numpy/)
- [NetworkX 3.0](https://networkx.org)
- [Matplotlib 3.7](https://matplotlib.org/stable/index.html)
- [SciPy 1.10.1](https://scipy.org)
- [Pandas 1.5.3](https://pandas.pydata.org/docs/user_guide/index.html)
- [Python-Louvain 0.16](https://python-louvain.readthedocs.io/en/latest/api.html#indices-and-tables)
- [Keras 2.12.0](https://keras.io/api/)

## An overview of the architecture
```mermaid
<style>
  .styleClass > rect {
    fill: #F0FFFF;
    stroke: #ffff00;
    stroke-width: 4px;
  }
</style>
classDiagram
  class Client:::styleClass
  DirectionL
  TrainingSetBuilder<..Client
  Client..>NNCF
  URMManager<..Client	
  
  NNCF..>EmbeddingBuilder
  NNCF..>NNFF
  NNCF..>NeighborhoodsBuilder  
  NNCF..>Learning
  
  
  Learning..>NNFF
  Learning..>BackPropagation
  
  NNFF..>ActivationFunctions
  
  BackPropagation..>ErrorFunctions
  BackPropagation..>ActivationFunctions

```

## References
T. Bai, J.-R. Wen, J. Zhang and W. Xin Zhao, "A Neural Collaborative Filtering Model with Interaction-based Neighborhood," in CIKM '17: Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, 2017.

A. Karatzoglou and B. Hidasi, "Deep Learning for Recommender Systems," in RecSys '17: Proceedings of the Eleventh ACM Conference on Recommender, New York, NY, United States, 2017. 

X. He, L. Liao, H. Zhang, L. Nie, X. Hu and T.-S. Chua, "Neural Collaborative Filtering," in WWW '17: Proceedings of the 26th International Conference on World Wide Web, 2017. 

C. C. Aggarwal, in Recommender Systems: The Textbook, Springer, 2016.

F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context.," in ACM Transactions on Interactive Intelligent Systems (TiiS), 2015. 
