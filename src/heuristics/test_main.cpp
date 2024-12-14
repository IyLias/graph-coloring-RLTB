
#include "gcp_solver.h"
#include <fstream>
#include <sstream>

using namespace std;

vector<vector<int>> readDIMACS(const string& filename){

 ifstream file(filename);
 if (!file.is_open()){
    throw runtime_error("Could not open file");
 }

 string line;
 int nodes=0, edges=0;
 vector<vector<int>> graph;

 while(getline(file, line)){
     if (line.empty() || line[0] == 'c'){
	 // ignore comments and empty lines	 
	 continue;
     }

     istringstream iss(line);
     char type;
     iss >> type;
   
     if (type == 'p'){
	string format;
	iss >> format >> nodes >> edges;
	graph.resize(nodes, vector<int>(nodes, 0));
     } else if (type == 'e'){
	int u,v;
	iss >> u >> v;
	u-=1;
	v-=1;
	graph[u][v] = 1;
	graph[v][u] = 1;
     }

 } // end of while loop

 file.close();
 return graph;

}

vector<int> generate_solution(int nodes, int k) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, k - 1);

    vector<int> solution(nodes,0);

    for (int i = 0; i < nodes; ++i) {
        solution[i] = dis(gen);
    }

    return solution;
}


int main(int argc, char* argv[]){

 if (argc < 3){ 
   cerr << "Usage: " << argv[0] << "<DIMACS file> <k(numColors)>" << endl;
   return -1;
 }

 string filename = argv[1];
 vector<vector<int>> graph;

 try{
    graph = readDIMACS(filename);
 } catch (const exception& e){
    cerr << "ERROR when reading graph: " << e.what() << endl;
    return -1;
 }
 
 cout << "Graph node: " << graph.size() << endl;


 int k = stoi(argv[2]);
 GCP_Solver solver(graph, k, "tabucol");
 vector<int> solution = generate_solution(graph.size() ,k);

 int reward = solver.solve(solution);
 cout << "Tabucol reward: " << reward << endl;


 return 0;
}
