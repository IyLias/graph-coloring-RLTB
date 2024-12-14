import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt

import random
import tyro
import torch
import time

from heuristics.gcp_solver import GCP_Solver



class GCP_Env(gym.Env):

    metadata = {"render_modes": ["human","file","console","rgb_array"], "render_fps": 10}

    def __init__(self,graph, k, total_steps, render_mode="console", convergence_steps=512, convergence_threshold=5):
        super().__init__()

        # total number of episodes
        self.n_episode = 0
        self.n_step = 0
        self.total_steps = 0
        
        # total timesteps for RL
        self.global_total_steps = total_steps

        # Given one Graph is the Environment of Agent
        self.graph = graph
        self.N = self.graph.order()
        self.size = self.graph.size()
        self.k = k
        

        self.setup_graph()

        # action is [i,j] tuple with i:node and j:color
        #self.action_space = spaces.Tuple((spaces.Discrete(self.N), spaces.Discrete(self.k)))
        self.action_space = spaces.Discrete(self.N * self.k)

        total_features = 3 + self.N * 4 + self.k * 3 + 1
        self.observation_space = spaces.Box(low=0, high=self.N, shape=(total_features,),dtype=np.float32)
        
        self.node_feature_size = 4
        self.color_feature_size = 3

        # solution is [N] array with color values in each node
        self.solution_space = spaces.Box(low=0, high=k - 1, shape=(self.N,), dtype=np.int32)

        self.state = None
        
        # conflict list for convergence test
        self.conflict_history = []
        self.convergence_steps = convergence_steps
        self.convergence_threshold = convergence_threshold

        # data for conflict zero chart
        self.num_zero_conflicts = 0


        # tabucol solver
        self.tabucol_solver = GCP_Solver(
                self.adj_matrix,
                k,
                "pg_tabucol"
        )

        assert render_mode is not None, "render mode should be designated"
        self.render_mode = render_mode
        
        # render properties setting
        self.color_map = None
        self.layout = nx.spring_layout(self.graph,k=0.1)
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        plt.ion()

        # reset 
        self.reset()
        
        # time when beginning
        self.start_time = time.time()
        self.end_time = self.start_time
    
    def setup_graph(self):
        """ 
            setup_graph() function organizes graph data and generates adj matrix of graph    
        
        """
        
        self.adj_matrix = nx.to_numpy_array(self.graph,dtype=int)
        self.adj_matrix = self.adj_matrix.tolist()
        



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # reset number of steps 
        self.n_step = 0
        self.n_episode += 1

        # reset reward setting
        self.n_reward = 0
        self.mean_reward = 0.0
        self.sum_squares = 0.0

        # conflict setting 
        self.conflict_history = []
        

        self.solution_space = spaces.Box(
            low=0, high=self.k-1, shape=(self.N,), dtype=np.int32, seed=seed
        )
        #self.action_space = spaces.MultiDiscrete([self.N, self.k], seed=seed)
        self.action_space = spaces.Discrete(self.N * self.k)

        #self.solution = self.solution_space.sample()
        self.solution = np.random.randint(0,self.k, size=self.N)
        self.n_colors_used = len(set(self.solution))
        
        # initialize info data
        self.initialize_obs()
        self.n_conflicts = self.calculate_conflicts()
        self.is_found_proper_coloring = False
        self.cur_reward = 0
        self.prev_action = 0
        
        print(f"New EPISODE Conflicts: {self.n_conflicts}")

        observation = self._get_obs()
        info = self._get_info() 

        return observation, info


    
    def _get_obs(self):
        """ Return observation

            Returns:
                graph feature, node feature, color feature and k
        """
        
        graph_features_flat = np.ravel(self.graph_feats)
        node_features_flat = np.ravel(self.node_feats)
        color_features_flat = np.ravel(self.color_feats)
        k_feature = np.array([self.k], dtype=np.float32)

        features = np.concatenate((graph_features_flat, node_features_flat,color_features_flat, k_feature))
        """features = {
            "graph_features": self.graph_feats,
            "node_features": self.node_feats,
            "color_features": self.color_feats,
            "k": self.k
        }"""

        return features



    def _get_info(self):
        """ Return info 

            Returns:
                current score, number of steps taken, solution and action
        """
        return {
                "reward": self.cur_reward,
                "num_colors_used": self.n_colors_used,
                "length": self.n_step,
                "number_of_conflicts": self.n_conflicts,
                "found_proper_coloring": self.is_found_proper_coloring,
                "num_zero_conflicts" : self.num_zero_conflicts,
                "solution": self.solution,
                "previous_action": self.prev_action
        }


    


    def initialize_obs(self):
        
        # Features of graph: (maximum degree, minimum degree, k)
        graph_feats = np.zeros((3,),dtype=np.float32)
        
        # Features of node: (degree, degree centrality, conflicting edges, neighboring groups)
        node_feats = np.zeros((self.N, 4), dtype=np.float32)
        # Features of color: (conflicts driven by color, total number of nodes in group j, percentage of this color is used)
        color_feats = np.zeros((self.k, 3), dtype=np.float32)

        # initialize graph_feats
        degrees = dict(self.graph.degree())
        max_degree = max(degrees.values()) if degrees else 0
        min_degree = min(degrees.values()) if degrees else 0

        graph_feats = np.array([max_degree, min_degree, self.k], dtype=np.float32)
        
        # initialize node_feats
        for node in self.graph.nodes():
            conflicts = 0
            groups = set()
            for adj in self.graph.neighbors(node):
                if self.solution[node] == self.solution[adj]:
                    conflicts += 1
                groups.add(self.solution[adj])
            node_feats[node, :] = [self.graph.degree(node), self.graph.degree(node)/max_degree, conflicts, len(groups)]

        # initialize col_feats
        nodes_with_color = np.zeros((self.k), dtype=np.int32)
        for col in self.solution:
            nodes_with_color[col] += 1
                    
        conflicts_with_color = np.zeros((self.k), dtype=np.int32)
        for node in self.graph.nodes():
            for adj in self.graph.neighbors(node):
                if self.solution[adj] == self.solution[node]:
                    color = self.solution[node]
                    conflicts_with_color[color] += 1
       
        for col in self.solution:
            color_feats[col,:] = [conflicts_with_color[col]//2, nodes_with_color[col], nodes_with_color[col]/self.graph.order()]

        
        self.graph_feats = graph_feats
        self.node_feats = node_feats
        self.color_feats = color_feats

    



    def step(self, action):
        
        # Given action as a integer number, we extract promising node and color 
        action_node = action // self.k
        action_color = action % self.k
    
        self.solution[action_node] = action_color
        
        prev_color = self.solution[action_node]
        self.cur_reward = -0.0001
        max_reward = self.size
        del_conflicts = 0
        del_colors_used = 0
        
        self.is_found_proper_coloring = False

        if prev_color != action_color:
            conflicts = self.node_feats[action_node, 1]
            n_colors_used = len(set(self.solution))

            new_conflicts = 0
            neighbors_group = set()
            for neighbor in self.graph.neighbors(action_node):
                if self.solution[action_node] == self.solution[neighbor]:
                    new_conflicts += 1

                neighbors_group.add(self.solution[neighbor])

                if self.solution[neighbor] == prev_color:
                    self.node_feats[neighbor, 2] -= 1
                if self.solution[neighbor] == action_color:
                    self.node_feats[neighbor, 2] += 1

            # Update number of conflicts of node
            self.node_feats[action_node, 2] = new_conflicts
            # Update number of neighbor groups of node
            self.node_feats[action_node, 3] = len(neighbors_group)

            # Update ratio of new_group 
            nodes_with_color = np.zeros((self.k), dtype=np.int32)
            for col in self.solution:
                nodes_with_color[col] += 1
                    
            conflicts_with_color = np.zeros((self.k), dtype=np.int32)
            for node in self.graph.nodes():
                for adj in self.graph.neighbors(node):
                    if self.solution[adj] == self.solution[node]:
                        color = self.solution[node]
                        conflicts_with_color[color] += 1
       
            self.color_feats[prev_color,:] = [conflicts_with_color[prev_color]//2, nodes_with_color[prev_color], nodes_with_color[prev_color] / self.graph.order()]
            self.color_feats[action_color,:] = [conflicts_with_color[action_color]//2, nodes_with_color[action_color], nodes_with_color[action_color] / self.graph.order()]


            # 2 factors: # of conflicts and # of colors used 
            del_conflicts = conflicts - new_conflicts
            del_colors_used = self.n_colors_used - n_colors_used 
            
            # reward = lambda*(f(s)-f(s')) + mu*(C(s)-C(s'))
            _lambda = 0.01
            _mu = 0
            self.cur_reward = _lambda * del_conflicts + _mu * del_colors_used 
        
            # add conflict 
            self.conflict_history.append(new_conflicts)


        self.n_conflicts = self.calculate_conflicts() 

        terminated = False
        max_steps_til_heuristics = 64
    
        self.total_steps += 1
        self.n_step += 1
        if self.n_conflicts == 0 or self.is_conflict_convergent() == True:
            terminated = True
            
        self.prev_action = action

        # after x steps of RL agent, take tabucol algorithm for y steps
        if self.total_steps % max_steps_til_heuristics == 0:
            
            rl_prob = np.exp(-1.5 * self.total_steps / self.global_total_steps)
            if np.random.rand() > rl_prob:
                #print("tabucol executing..")
                tabucol_solution, tabucol_reward = self.tabucol_solver.solve(self.solution)
                self.solution = tabucol_solution
                self.n_conflicts = self.calculate_conflicts()
                if self.n_conflicts == 0:
                   print("found proper coloring from tabucol")
                self.cur_reward += float(0.1*tabucol_reward)    
            
            # render for every tabucol steps
            self.render()

        if self.n_conflicts == 0:
            self.is_found_proper_coloring = True
            self.num_zero_conflicts += 1
            if self.num_zero_conflicts == 1:
                self.end_time = time.time()    

            self.cur_reward = max_reward
            self.n_colors_used = len(set(self.solution))   
            print("found proper coloring")
            print(f"number of used color: {self.n_colors_used}") 
            

        observation = self._get_obs()
        info = self._get_info()

        return observation, self.cur_reward, terminated, False, info
        


    def render(self):

        if self.render_mode == "console":
            self.graph.print_colorings()
            return
        
        def is_in_ipython():
            try:
                __IPYTHON__
                return True
            except NameError:
                return False

        if is_in_ipython():
            from IPython.display import clear_output
            # clear_output(wait=True)


        if self.color_map is None:
            self.color_map = dict(
                [
                    (
                        j,
                        f"#{''.join([random.choice('0123456789ABCDEF') for i in range(6)])}",
                    )
                    for j in range(self.k)
                ]
            )

        if self.layout is None:
            self.layout = nx.spring_layout(self.graph)
        
        alphas = [
            1.0 if self.solution[x] == self.solution[y] else 0.1
            for x, y in nx.edges(self.graph)
        ]
        
        node_colors = [self.color_map[node] for node in self.solution]

        edge_colors = [
            "#ff0000" if self.solution[x] == self.solution[y] else "#000000"
            for x, y in nx.edges(self.graph)
        ]
        
        self.ax.clear()
        self.ax.axis('off')

        nx.draw_networkx_nodes(self.graph, self.layout, node_color=node_colors)
        nx.draw_networkx_labels(self.graph, self.layout)
        nx.draw_networkx_edges(
            self.graph, self.layout, edge_color=edge_colors, alpha=alphas
        )
        
        self.n_colors_used = len(set(self.solution))

        # Display these info as text on the plot
        info_text = f"Graph order: {self.graph.order()}\nGraph size: {self.graph.size()}\nNumber of Colors: {self.n_colors_used} \nConflicts: {self.n_conflicts} \nElapsed Time: {self.end_time-self.start_time} \nColors used for solving: {self.n_colors_used}"
        self.ax.text(0.01, 0.99, info_text, transform=self.ax.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            plt.show()
            
            # if capture_video, should return pixel array
            if self.render_mode == "rgb_array":
                self.fig.canvas.draw()
                image_from_plot = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                return image_from_plot
        


    def is_conflict_convergent(self):

        if len(self.conflict_history) > self.convergence_steps:
            self.conflict_history.pop(0) # remove first element
            conflict_changes = [abs(self.conflict_history[i] - self.conflict_history[i-1]) for i in range(1, len(self.conflict_history))]
            if all(change <= self.convergence_threshold for change in conflict_changes):
                return True
            else:
                return False

        else:
            return False




    def close(self):
        pass

    def calculate_conflicts(self):
        """ Return score: total number of conflicts

            Returns:
                number of conflicts value
        """

        conflicts = 0
        for node in self.graph.nodes():
            for neighbor in self.graph.neighbors(node):
                if self.solution[node] == self.solution[neighbor]:
                    conflicts += 1

        conflicts = conflicts // 2

        #n_conflicts = sum(1 for (u,v) in self.graph.edges() if self.solution[u] == self.solution[v])    
        return conflicts


    def _get_graph(self):
        return self.graph

    def _get_num_nodes(self):
        return self.N


    def _get_num_colors(self):
        return self.k



