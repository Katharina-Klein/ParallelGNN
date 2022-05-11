#include "vertex.h"
#include "graph_distributed.h"
#include <iostream>
#include <fstream>
#include <string>

#include "bulk/bulk.hpp"

#ifndef GNN_DISTRIBUTED_H
    #define GNN_DISTRIBUTED_H
    class gnn_distributed
    {
    public:
        gnn_distributed();
        gnn_distributed(int T, graph_distributed *input_graph, std::vector<Matrix*> weights);
        ~gnn_distributed();

        void forward_pass(bulk::world& world);


    private:

        graph_distributed *input_graph;
        int dimension;
        int num_timesteps;

        std::vector<Matrix*> weights;

        void communicate_states(bulk::world& world);


    protected:



    };


#endif // GNN_DISTRIBUTED_H


