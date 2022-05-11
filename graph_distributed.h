#include <vector>
#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <string>

    typedef Eigen::MatrixXf Matrix;
    typedef Eigen::RowVectorXf RowVector;
    typedef Eigen::VectorXf ColVector;
    typedef Eigen::Vector2i Edge;

#include "bulk/bulk.hpp"

#include "vertex.h"

#ifndef GRAPH_DISTRIBUTED_H
    #define GRAPH_DISTRIBUTED_H
    class graph_distributed
    {

    public:
        graph_distributed();
        graph_distributed(int d, std::string filename_vertices, std::string filename_edges, bulk::world& world);
        ~graph_distributed();


        int num_vertices;
        int num_halo_vertices;
        int dimension;

        std::vector<vertex*> vertices;
        std::vector<int> requested_vertices;
        std::vector<int> global_vertex_indices;
        std::vector<int> local_vertex_indices;
        std::vector<int> halo_vertices;
        std::vector<Edge*> halo_edges;




    private:
        void build_graph_vertices(std::string filename, bulk::world& world);
        void build_graph_edges(std::string filename);
        void build_halo(std::string filename);




    protected:





    };



#endif // GRAPH_DISTRIBUTED_H

