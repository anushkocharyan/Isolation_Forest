# Isolation_Forest

The goal for this project is to implement the (Isolation Forest algorithm)[http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.673.5779&rep=rep1&type=pdf] by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou for anomaly detection. 

The Isolation Forest algorithm focuses on anomalies instead of modeling normal observations and is based on the idea that anomalies are more susceptible to isolation since they are 'few and different'. The algorithm is based on decision trees, like any other tree ensemble method. Trees are split recursively on a random value within the range of a random feature until all observations are isolated.  This random partitioning produces significantly shorter paths for anomalies. Thus, when a forest of trees produce short path lengths for certain observations these observations are highly likely to be anomalies.

The data used in this project (creditcard.csv) can be downloaded (here)[https://www.kaggle.com/mlg-ulb/creditcardfraud].

