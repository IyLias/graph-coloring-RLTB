#ifndef GCP_SOLVER_H
#define GCP_SOLVER_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <random>
#include <ctime>
#include <cstdlib>
#include <deque>

#define MAX_NUM	12345678


using namespace std;

struct pair_hash{
    template <class T1, class T2>
    size_t operator() (const pair<T1,T2>& pair) const{
	    return hash<T1>()(pair.first) ^ hash<T2>()(pair.second);
    }
};


class GCP_Solver{

 public:

   GCP_Solver(const vector<vector<int>>& target_graph, int k, string algorithm="pg_tabucol");

   std::pair<vector<int>, int> solve(const vector<int>& solution);


 private:

   std::pair<vector<int>, int> tabucol(const vector<int>& solution, int k, int tabu_size=7, int reps=256, int max_iterations=2500, float alpha=0.6);

   std::pair<vector<int>, int> position_guided_tabucol(const vector<int>& solution, int k, int tabu_size=15, int reps=256, int max_iterations=2500, float alpha=0.6);

   int get_distance(const vector<int>& coloring1, const vector<int>& coloring2);

   bool is_already_visited(const vector<int>& solution);

   int count_conflicts(const vector<int>& solution, int node=-1);

   int choose_color(const vector<int>& solution, int node, float p=0.8);

   void print_graph();
   
   void print_solution(const vector<int>& solution);

   int nodes;
   int k;
   int R;

   vector<vector<int>> graph;

   string algorithm;
   
   // tabu list for original tabu search
   unordered_map<pair<int,int>,int,pair_hash> tabu_list;
   
   // tabu list for PGTS
   deque<pair<vector<int>, int>> pg_tabu_list;

   vector<vector<int>> recorded_solutions;
};

#endif



