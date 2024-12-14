# GCP-RLTB

This repository presents a hybrid approach combining Reinforcement Learning (RL) with the Tabucol, which is a version of tabu search specifically designed for the Graph Coloring Problem (GCP), enhanced by Graph Neural Networks (GNNs), to tackle the Graph Coloring Problem(GCP).

<br>

## Graph Coloring Problem (GCP)

Let $G$ be an undirected graph, $G=(V,E)$ with vertex set $V$ and edge set $E$. Now we define a mapping function $f$, where $f: V$ $\mapsto$ \{1,2,..., k \}. Then the value $f(v)$ of vertex $v$ is the color of $v$. If two adjacent vertices $x$ and $y$ have the same color $j$, then it is called a <strong> conflict </strong>. A coloring with no conflicts is called a proper-coloring. 

The chromatic number of $G$, denoted by $\chi(G)$, is the smallest $k$ for which there exists a $k$-coloring of $G$. Based on these definitions, GCP is the problem of finding the chromatic number of given graph $G$, which is the minimum number of colors required to color the vertices of $G$ such that no two adjacent vertices share the same color.


<br><br>
