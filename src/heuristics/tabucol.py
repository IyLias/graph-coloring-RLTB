from collections import deque
from random import randrange
import networkx as nx 


def tabucol(graph,k,tabu_size=7,reps=100,max_iterations=10000, alpha=0.6):
    
    """ 
    function tabucol() implements tabucol algorithm
        
        parameter 

            graph: networkx graph  
            k : number of colors using 
            tabu_size: size of tabu list 
            reps: number of iterations searching neighbor solution
            max_iterations: maximum iterations of tabucol
            alpha: parameter of duration equation

        return 

            solution: solution list applied tabucol
    """
    N = graph.order()
    colors = list(range(k))
    
    # Tabu List 
    tabu = {}
    iterations = 0
    
    # Generate Initial solution
    solution = {i: colors[randrange(0, len(colors))] for i in range(N)}

    conflicts = 0
    aspiration_level = dict()
    while iterations < max_iterations:
        
        conflicts = count_conflicts(graph, solution)
        conflicting_nodes = set()
        for node in graph.nodes():
            for neighbor in graph.neighbors(node):
                if solution[node] == solution[neighbor]:
                    conflicting_nodes.add(node)
                    conflicting_nodes.add(neighbor)

        conflicting_nodes = list(conflicting_nodes)
        # Exit Condition
        if conflicts == 0:
            break

        
        # Exploring Neighbor solutions
        new_solution = None
        found_better = False
        for r in range(reps):
            node = conflicting_nodes[randrange(0,len(conflicting_nodes))]
            
            # allocate best color: generating minimum conflicts
            new_color = choose_color(graph, solution, node, colors)            
            new_solution = solution.copy()
            new_solution[node] = new_color

            new_conflicts = count_conflicts(graph, new_solution)

            L = randint(0,9)
            # duration = L + f(s)*alpha
            duration = int(L + new_conflicts * alpha) 

            # If found a better solution, 
            if new_conflicts < conflicts:
                found_better = True

                # if f(s') <= A(f(s))
                if new_conflicts <= aspiration_level.setdefault(conflicts, conflicts-1):
                    # set A(f(s)) = f(s')-1
                    aspiration_level[conflicts] = new_conflicts - 1
                    
                    if (node, new_color) in tabu and tabu[(node, new_color)] > iterations:
                        tabu.pop((node, new_color))
                    else:
                        tabu[(node, new_color)] = iterations + duration
                
                # case for found good solution but that action is in tabu, then should search more
                else:
                    if (node, new_color) in tabu:
                        continue
                break
        

        if found_better:
            solution = new_solution
        
        # Clean up expired tabu entries
        tabu = {move: expiry for move, expiry in tabu.items() if expiry > iterations}

        iterations += 1
        if iterations % 500 == 0:
            print(f"Iteration: {iterations}, Conflicts: {conflicts}")

    
    # After all iterations, return solution
    return solution



def count_conflicts(graph, solution, node=None):
    """
    function count_conflicts() calculates number of conflicts given graph and solution. 

        parameter 
            
            graph: networkx graph 

            solution: 

            node: if node is given, only search conflicts around the node


        return 
            
            conflicts: number of conflicts

    """

    conflicts = 0
    if node is None:
        for node in graph.nodes():
            for neighbor in graph.neighbors(node):
                if solution[node] == solution[neighbor]:
                    conflicts += 1

    else:
        for neighbor in graph.neighbors(node):
            if solution[node] == solution[neighbor]:
                conflicts += 1
        
    return conflicts




def choose_color(graph, solution, node, colors):
    """
    function choose_color() chooses best color given target node. 
    Best color means the color which makes minimal conflicts 
        
        parameter

            graph: networkx graph

            solution:

            node: target node

            colors: color array

        
        return 

            best_color

    """

    min_conflicts = float('inf')
    best_color = solution[node]
    for color in colors:
        if color != solution[node]:
            solution[node] = color
            conflicts = count_conflicts(graph, solution,node)
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_color = color

    return best_color

            




