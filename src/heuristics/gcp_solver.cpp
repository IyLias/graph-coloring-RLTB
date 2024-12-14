
#include "gcp_solver.h"


GCP_Solver::GCP_Solver(const vector<vector<int>>& target_graph, int k, string algorithm)
: k(k), algorithm(algorithm){
    this->nodes = target_graph.size();
    //printf("constructor Node: %d\n",this->nodes);
    this->R = (int)(0.1*this->nodes);

    this->graph.resize(nodes, vector<int>(nodes, 0));
    this->graph = target_graph; 


    std::srand(std::time(nullptr)); // initialize random seed
}

void GCP_Solver::print_graph(){


    printf("print graph started\n");
    for(int u=0;u<this->nodes;u++){
        for(int v=0;v<this->nodes;v++)
            printf("%d ",this->graph[u][v]);

        printf("\n");
    }
}


void GCP_Solver::print_solution(const vector<int>& solution){

    for(int n=0;n<this->nodes;n++)
 	printf("%d ",solution[n]);
    printf("\n");

}



std::pair<vector<int>,int> GCP_Solver::solve(const vector<int>& solution){

    int reward=0;
    if (algorithm == "tabucol")
	//printf("tabucol starting..\n");
    	return tabucol(solution, k); 
    else if(algorithm == "pg_tabucol")
	printf("pgts starting..\n");
        return position_guided_tabucol(solution,k);

    return {solution, reward};
}


std::pair<vector<int>,int> GCP_Solver::tabucol(const vector<int>& solution, int k, int tabu_size, int reps, int max_iterations, float alpha){

    /*
        function tabucol() implements tabucol algorithm

            parameter

                solution: initial solution
                k : number of colors using
                tabu_size: size of tabu list
                reps: number of iterations searching neighbor solution
                max_iterations: maximum iterations of tabucol
                alpha: parameter of duration equation

            return

                solution: final solution applied tabucol
    */
    int initial_conflicts = count_conflicts(solution);
    printf("Iterations: 0, Conflicts: %d\n",initial_conflicts);
    	
    vector<int> result_solution(nodes,0);
    result_solution = solution;

    // For Random Values
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> nodeDis(0, nodes-1);
    uniform_int_distribution<> randDis(0, 8);    


    int iterations = 0;
    int conflicts = 0;
    unordered_map<int, int> aspiration_level;

    while (iterations < max_iterations){
    
	conflicts = count_conflicts(result_solution);
	unordered_set<int> conflicting_nodes;
	for (int u=0; u<nodes; u++){
	    for (int v=u+1; v<nodes; v++){
		if(graph[u][v] && (result_solution[u] == result_solution[v])){
		   conflicting_nodes.insert(u);
		   conflicting_nodes.insert(v);
		}
            }

	}

	// Exit Condition
	if (conflicts == 0)
           break;

	vector<int> conflicting_nodes_list(conflicting_nodes.begin(), conflicting_nodes.end());
    	uniform_int_distribution<> conf_nodeDis(0, conflicting_nodes_list.size()-1);
	vector<int> new_solution(nodes,0);

	for (int r=0; r < reps; r++){

	   int node = conflicting_nodes_list[conf_nodeDis(gen)];

	   int new_color = choose_color(result_solution, node);
	   new_solution = result_solution;
	   new_solution[node] = new_color;

	   int new_conflicts = count_conflicts(new_solution);
	   
	   int L = randDis(gen);
	   int duration = int(L + new_conflicts * alpha);

	   if (new_conflicts < conflicts){
	      
	      if (new_conflicts <= aspiration_level[conflicts]){
		  aspiration_level[conflicts] = new_conflicts - 1;
		  auto tabu_entry = tabu_list.find({node, new_color});
		  if (tabu_entry != tabu_list.end() && tabu_entry->second > iterations)
		      tabu_list.erase(tabu_entry);
		  else
	  	      tabu_list[{node,new_color}] = iterations + duration;
	      
              // case for found good solution but that action is in tabu, then should search more
	      }else{
		  
		  if (tabu_list.find({node, new_color}) != tabu_list.end())
		      continue;
		  else
 		      tabu_list[{node, new_color}] = iterations + duration;
              }

	      break;
	   }

    	}  
  
	result_solution = new_solution; 

 	iterations += 1;
    	//if (iterations % 500 == 0){
        //    printf("Iterations: %d, Conflicts: %d\n",iterations, conflicts);
    	//}	

	// Clean up expired tabu entries
        for (auto it = tabu_list.begin(); it != tabu_list.end();) {
            if (it->second <= iterations) {
                it = tabu_list.erase(it);
            } else {
                ++it;
            }
        }
    }

    // After while loop, calculate reward
    int final_conflicts = count_conflicts(result_solution);    
    int tabucol_reward = initial_conflicts - final_conflicts;
	
    printf("final conflict: %d\n", final_conflicts);    

    // After all iterations, return reward
    return {result_solution, tabucol_reward};

}




std::pair<vector<int>, int> GCP_Solver::position_guided_tabucol(const vector<int>& solution, int k,int tabu_size, int reps, int max_iterations, float alpha){

    int initial_conflicts = count_conflicts(solution);
    printf("Iterations: 0, Conflicts: %d\n",initial_conflicts);
    	
    vector<int> result_solution(nodes,0);
    result_solution = solution;

    vector<int> pivot_solution(nodes,0);
    pivot_solution = solution;

    // For Random Values
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> nodeDis(0, nodes-1);
    uniform_int_distribution<> randDis(0, 15);    


    int iterations = 0;
    int conflicts = 0;

    while (iterations < max_iterations){
    
	conflicts = count_conflicts(result_solution);
	unordered_set<int> conflicting_nodes;
	for (int u=0; u<nodes; u++){
	    for (int v=u+1; v<nodes; v++){
		if(graph[u][v] && (result_solution[u] == result_solution[v])){
		   conflicting_nodes.insert(u);
		   conflicting_nodes.insert(v);
		}
            }

	}

	// Exit Condition
	if (conflicts == 0)
           break;

	vector<int> conflicting_nodes_list(conflicting_nodes.begin(), conflicting_nodes.end());
    	uniform_int_distribution<> conf_nodeDis(0, conflicting_nodes_list.size()-1);
	vector<int> new_solution(nodes,0);

	for (int r=0; r < reps; r++){

	   int node = conflicting_nodes_list[conf_nodeDis(gen)];

	   int new_color = choose_color(result_solution, node);
	   new_solution = result_solution;
	   new_solution[node] = new_color;

	   int new_conflicts = count_conflicts(new_solution);
	   
	   int L = randDis(gen);
	   int duration = int(L + new_conflicts * alpha);
           
	   if (new_conflicts < conflicts){
		
	      // check current new solution is in pg tabu list
	      auto it = std::find_if(pg_tabu_list.begin(), pg_tabu_list.end(),
			      	    [&new_solution](const pair<vector<int>,int>& tabu_item){
				    		return tabu_item.first == new_solution;
				    });
	      bool is_in_tabu = (it != pg_tabu_list.end());
	      if (is_in_tabu == true)
		 continue;
	      else{

	         if (get_distance(new_solution, pivot_solution) > this->R){

		    pivot_solution = new_solution;
 	  	    if (is_already_visited(pivot_solution) == false){
		       recorded_solutions.push_back(pivot_solution);
		    }
		    
	         }   

	         pg_tabu_list.push_back({new_solution,duration});

	         int pivot_conflicts = count_conflicts(pivot_solution);
	         if(new_conflicts < pivot_conflicts)
		    pivot_solution = new_solution;

	         break;
	      }
	   }

    	}  
  
	result_solution = new_solution; 

 	iterations += 1;

	// Clean up expired tabu entries
        for (auto it = pg_tabu_list.begin(); it != pg_tabu_list.end();) {
            if (it->second <= iterations) {
                it = pg_tabu_list.erase(it);
            } else {
                ++it;
            }
        }
    }

    // After while loop, calculate reward
    int final_conflicts = count_conflicts(result_solution);    
    int tabucol_reward = initial_conflicts - final_conflicts;
	
    printf("final conflict: %d\n", final_conflicts);    

    // After all iterations, return reward
    return {result_solution, tabucol_reward};
}

int GCP_Solver::get_distance(const vector<int>& coloring1, const vector<int>& coloring2){

  int dist=0;
  int N = (int)coloring1.size();
  
  for(int i=0;i<N;i++)
     if(coloring1[i] != coloring2[i])
	 dist++;    
 
  return dist;
}



bool GCP_Solver::is_already_visited(const vector<int>& solution){

   for(auto coloring : this->recorded_solutions){
      if(get_distance(solution, coloring) <= this->R)
	  return true; 
   }

   return false;
}



int GCP_Solver::count_conflicts(const vector<int>& solution, int node){
    /*
        function count_conflicts() calculates number of conflicts given graph and solution.

            parameter

                graph: networkx graph

                solution:

                node: if node is given, only search conflicts around the node


            return

                conflicts: number of conflicts

    */
	
    int conflicts = 0;
    if(node == -1){
        for (int u=0;u<this->nodes;u++){
	    for (int v = u+1; v<this->nodes;v++){
		  if(this->graph[u][v] && (solution[u] == solution[v])){
			conflicts += 1;
		  }
	    }
	}
	

    }else{

	for (int v=0; v<this->nodes;v++){
	    if (this->graph[node][v] && (solution[node] == solution[v])){
		  conflicts += 1;
	    }
	}
    }
    

    return conflicts;
}




int GCP_Solver::choose_color(const vector<int>& solution, int node, float p){

    /*
        function choose_color() chooses best color given target node.
        Best color means the color which makes minimal conflicts

            parameter

                graph: networkx graph

                solution:

                node: target node

                colors: color array


            return

                best_color

    */

    int min_conflicts = MAX_NUM;
    int best_color = solution[node];
    
    vector<int> colors(k);
    iota(colors.begin(), colors.end(), 0);    
    
    double rand_prob = static_cast<double>(std::rand()) / RAND_MAX;
    bool choose_random = (rand_prob > p);

    if (choose_random){
      vector<int> available_colors;
      for (int color : colors){
 	if (color != solution[node])
	   available_colors.push_back(color);
      }
	
      if (!available_colors.empty()){
	  int rand_idx = std::rand() % available_colors.size();
	  return available_colors[rand_idx];
      } else
	  return best_color;


    } else {
	// choose most promising color (that decreases conflicts most)
	
    	for (int color : colors){
	   if (color != solution[node]){
	   	vector<int> temp_solution = solution;
	   	temp_solution[node] = color;
	   	int conflicts = count_conflicts(temp_solution, node);
	   	if (conflicts < min_conflicts){
		    min_conflicts = conflicts;
		    best_color = color;
	   	}
           }

        }

    }

    return best_color;


}



