# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm,ReLU, Sequential, Dropout, BatchNorm1d
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import GCNConv, GraphNorm

import networkx as nx

from gym_snakegame.envs import SnakeGameEnv
from gym_gcp.envs import GCP_Env

from util import readDIMACS_graph
import math

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
   
    graph_file: str = ""
    """graph data file path which uses DIMACS format """
    
    graph_types: int = 0
    """types of training graph: 0 for random graph by Erdos-Renyi model, 1 for empty graph, 2 for complete graph, 3 for petersen_graph, 4 for flat graph, 5 for le450 graph, 6 for queen graph, 7 for myciel graph, 8 for games graph"""

    nodes: int = 100
    """the order of target graph"""
    probability: float = 0.5
    """probability of edge between nodes in target graph"""
    colors: int = 10
    """number of colors that can be used in target graph"""


def make_env(env_id, target_graph, k, seed, idx, total_steps, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, graph=target_graph, k=k, total_steps=total_steps,render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, graph=target_graph, k=k, total_steps=total_steps,render_mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


def setup_env(graph_file_path, graph_type, nodes, prob, ncolors, seed):
    graph = None

    if graph_file_path == "":
        # DSJCn.x(random graph: Erdos-Renyi model) 
        if graph_type == 0:
            graph = nx.gnp_random_graph(nodes, prob,seed=seed)
        elif graph_type == 1: # empty graph 
            graph = nx.empty_graph(n=nodes)
        elif graph_type == 2: # complete graph 
            graph = nx.complete_graph(n=nodes)
        elif graph_type == 3: # petersen graph 
            graph = nx.petersen_graph()
    else:
        # for case, graph data is given,
        graph = readDIMACS_graph(graph_file_path)



    max_degree = max(dict(graph.degree()).values())

    #clique_num = len(nx.approximation.max_clique(graph))
    
    # w(G) <= k <= Delta(G)
    #k = max(clique_num, min(max_degree+1, nodes))
    #print(f"Using k={k} for graph coloring (Clique Number: {clique_num}, Max degree: {max_degree})")
    
    k = ncolors
    edge_index = convert_to_edge_index(graph)
    edge_index = generate_edge_index(edge_index, nodes, k)
    print(edge_index.shape)

    return graph, k, edge_index





def set_run_name(graph_type, nodes, prob, ncolors, seed): 
    
    run_name = ""
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    #if args.random_graph == True:
    
    # DSJCn.x(random graph: Erdos-Renyi model) 
    if graph_type == 0:
       run_name = f"{args.env_id}__G({nodes},{prob})k{ncolors}__gnn__{int(time.time())}"
    elif graph_type == 1:
       run_name = f"{args.env_id}__O_{nodes}__k{ncolors}__gnn__{int(time.time())}"
    elif graph_type == 2:
       run_name = f"{args.env_id}__K_{nodes}__k{ncolors}__gnn__{int(time.time())}"
    elif graph_type == 3:
       run_name = f"{args.env_id}__Petersen__k{ncolors}__gnn__{int(time.time())}"
    elif graph_type == 4:
       run_name = f"{args.env_id}__flat{nodes}__k{ncolors}__gnn__{int(time.time())}"
    elif graph_type == 5:
       run_name = f"{args.env_id}__le450_25__k{ncolors}__gnn__{int(time.time())}"
    elif graph_type == 6:
       nodes = int(math.sqrt(nodes))
       run_name = f"{args.env_id}__queen{nodes}_{nodes}__k{ncolors}__gnn__{int(time.time())}"
    elif graph_type == 7: 
       run_name = f"{args.env_id}__myciel{nodes}__k{ncolors}__gnn__{int(time.time())}"
    elif graph_type == 8:
       run_name = f"{args.env_id}__games{nodes}__k{ncolors}__gnn__{int(time.time())}"
    


    return run_name



def convert_to_edge_index(G):
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    #print(edge_index.max().item())
    return edge_index


def get_feature_node_index(node, color, num_colors):
    return node * num_colors + color


def generate_edge_index(org_edge_index, num_nodes, num_colors):
    
    source_nodes = []
    target_nodes = []

    for node in range(num_nodes):
        for color_i in range(num_colors):
            for color_j in range(color_i+1, num_colors):
                idx_i = get_feature_node_index(node, color_i, num_colors)
                idx_j = get_feature_node_index(node, color_j, num_colors)

                source_nodes.extend([idx_i, idx_j])
                target_nodes.extend([idx_j, idx_i])


    for (u,v) in org_edge_index.t().tolist():
        for color in range(num_colors):
            idx_u = get_feature_node_index(u, color, num_colors)
            idx_v = get_feature_node_index(v, color, num_colors)

            source_nodes.extend([idx_u, idx_v])
            target_nodes.extend([idx_v, idx_u])

    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    return edge_index




# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env,nodes, colors, hidden_dim, num_layers):
        super().__init__()
         
        self.num_nodes = nodes 
        self.num_colors = colors

        self.node_feat_dim = 4
        self.color_feat_dim = 3
        
        self.combined_feat_dim = self.node_feat_dim + self.color_feat_dim
        
        self.gcn1 = GCNConv(self.combined_feat_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)


    def forward(self, flat_obs, edge_index):
        """
        Args:
            node_features: Tensor of shape [num_nodes, node_feature_dim]
            color_features: Tensor of shape [num_colors, color_feature_dim]
            edge_index: Tensor defining the graph connectivity
        Returns:
            q_values: Tensor of shape [1, num_nodes * num_colors]
        """

        # extract noded_features from observation
        node_features_start = 3
        node_features_end = node_features_start + self.num_nodes * self.node_feat_dim
        node_features = flat_obs[:,node_features_start:node_features_end].reshape(-1,self.num_nodes, self.node_feat_dim)
        
        # extract color_features from observation
        color_features_start = node_features_end
        color_features_end = color_features_start + self.num_colors * self.color_feat_dim
        color_features = flat_obs[:,color_features_start:color_features_end].reshape(-1,self.num_colors, self.color_feat_dim)
        

        # Expand node and color feature to create (node, color) pairs
        expanded_node_features = node_features.unsqueeze(2).repeat(1,1, self.num_colors,1)
        expanded_color_features = color_features.unsqueeze(1).repeat(1,self.num_nodes,1,1)

        combined_features = torch.cat([expanded_node_features, expanded_color_features], dim=3)
        combined_features = combined_features.view(-1, self.combined_feat_dim)
        
        # Apply GCN layers
        x = self.gcn1(combined_features, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)

        # Fully connected layers to produce Q-values
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
       
        batch_size = node_features.size(0)
        q_values = q_values.view(batch_size, self.num_nodes * self.num_colors)
        #print(q_values.shape)
        #print(q_values)

        return q_values


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    
    # set run file name 
    run_name = set_run_name(args.graph_types, args.nodes, args.probability, args.colors, args.seed)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(str(device) + " is used for training")
    
    # crucial parameters for GCP
    nodes = args.nodes
    probability = args.probability
    graph_type = args.graph_types
    ncolors = args.colors

    graph_file_path = args.graph_file

    graph, k, edge_index = setup_env(graph_file_path, graph_type, nodes, probability,ncolors, args.seed)
    
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, graph, k, args.seed + i, i, args.total_timesteps, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, nodes, ncolors, 16, 3).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, nodes, ncolors, 16, 3).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device), torch.Tensor(edge_index).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # record conflicts, colors_used, conflict_zero
        writer.add_scalar("charts/conflicts",infos["number_of_conflicts"],global_step)
        writer.add_scalar("charts/colors_used",infos["num_colors_used"],global_step)

        # check if any environment has zero conflicts 
        conflict_zero_accumulate = infos["num_zero_conflicts"]
        writer.add_scalar("charts/conflict_zero", conflict_zero_accumulate, global_step)  

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step},episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    #writer.add_scalar("chars/episodic_conflicts",info["episode"][""],global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)


        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations, edge_index.to(device)).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                
                # *** revise this part.. ***  
                data_actions = torch.argmax(data.actions, dim=1,keepdim=True)
                old_val = q_network(data.observations, edge_index.to(device)).gather(1, data_actions).squeeze()
                #print("td_target:",td_target.shape, "old_val: ",old_val.shape)
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    #print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
