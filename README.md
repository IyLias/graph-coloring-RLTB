# GCP-RLTB

This repository presents a hybrid approach combining Reinforcement Learning (RL) with the Tabucol, which is a version of tabu search specifically designed for the Graph Coloring Problem (GCP), enhanced by Graph Neural Networks (GNNs), to tackle the Graph Coloring Problem(GCP).

<br>

## Graph Coloring Problem (GCP)

Let $G$ be an undirected graph, $G=(V,E)$ with vertex set $V$ and edge set $E$. Now we define a mapping function $f$, where $f: V$ $\mapsto$ \{1,2,..., k \}. Then the value $f(v)$ of vertex $v$ is the color of $v$. If two adjacent vertices $x$ and $y$ have the same color $j$, then it is called a <strong> conflict </strong>. A coloring with no conflicts is called a proper-coloring. 

The chromatic number of $G$, denoted by $\chi(G)$, is the smallest $k$ for which there exists a $k$-coloring of $G$. Based on these definitions, GCP is the problem of finding the chromatic number of given graph $G$, which is the minimum number of colors required to color the vertices of $G$ such that no two adjacent vertices share the same color.

<br><br>


## Action Space

$[N,K]$ array with (vertex, color) pair

<br>

``` python3

self.action_space = spaces.Tuple((spaces.Discrete(self.N), spaces.Discrete(self.k)))

```

<br><br>


## Reward

The default reward for per step is -1e-4. The reward consists of 2 parts, RL reward and Heuristic(Tabucol) reward.

<br>

In the RL reward, we consider two main factors, number of conflicts and number of colors used. 

The reward function considering these two factors is as follows. 

$reward_{RL} = -\lambda * (f(s)-f(s')) + \mu*(C(s)-C(s'))$

<br>

``` python3
# 2 factors: # of conflicts and # of colors used
del_conflicts = conflicts - new_conflicts
del_colors_used = self.n_colors_used - n_colors_used

# reward = -lambda*(f(s)-f(s')) + mu*(C(s)-C(s'))
# We excluded color factor 
_lambda = 0.01
_mu = 0
self.cur_reward = -_lambda * del_conflicts + _mu * del_colors_used
```

<br>

In the Heuristic(Tabucol) reward, it's calculated by the difference between the number of conflicts after $x$ steps and $x+y$ steps. Thus the reward function is as follows. 

$reward_{HO} = f(s_x) - f(s_{x+y})$

The final reward considering these 2 rewards is as follows. 

$reward = reward_{RL} + reward_{HO}$


<br><br>





## Methodology 

### Deep Q-Network Architecture for Graph Coloring

The Q-Network processes these combined features through multiple layers, including GNN layers for message passing and feature aggregation, followed by fully connected layers. The final output layer produces Q-values corresponding to each (node, color) pair, representing the expected future rewards of selecting that specific action in the current state.


<br>

### Applying Graph Neural Network

The node and color features are concatenated to form comprehensive (node, color) pair representations. These combined features are then passed through multiple GNN layers, particularly Graph Convolutional Network (GCN) layers, within our Q-Network. In these GCN layers, message passing is performed to aggregate and update feature representations based on the graph's connectivity, effectively capturing both local and global structural information. This process results in an updated feature graph that integrates the nuanced interactions between nodes and colors.


<br><br>

## Architecture 

<img src="https://github.com/user-attachments/assets/a12863f6-777d-4e5f-95bf-a475abf3d298" width="900">



<br><br>

## Results 

(Update soon..)

<br><br>

## Code

Our project is primarily developed in Python, with the heuristic algorithm for Tabucol implemented in C++. 

The reinforcement learning framework is built using CleanRL and extensively utilizes PyTorch.



<br><br>


## Author 

Seok-Jin Kwon / IyLias
