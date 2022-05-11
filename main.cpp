

    #include <iostream>
    #include <ctime>

    #include <vector>
    #include <Eigen/Core>

        typedef Eigen::MatrixXf Matrix;
        typedef Eigen::RowVectorXf RowVector;
        typedef Eigen::VectorXf ColVector;


    #include "bulk/bulk.hpp"
    #include "set_backend.hpp"


    #include "vertex.h"
    #include "graph_distributed.h"
    #include "gnn_distributed.h"
    #include "helpers.h"




    int main(int argc,char* argv[])
    {

        environment env;
        int num_procs;
        if(argc > 1)
        {
            num_procs = std::__cxx11::stoi(argv[1]);
        }
        else
        {
            num_procs = env.available_processors();
        }

        env.spawn(num_procs, [argc, argv](bulk::world& world)
        {
            int s = world.rank();
            int p = world.active_processors();

            std::string filename1, filename2, filename3, filename4;
            int d, T;

            if(argc > 1)
            {
                filename1 = argv[4];
                filename2 = argv[5];
                filename3 = argv[6];
                filename4 = argv[7];

                d = std::__cxx11::stoi(argv[2]);
                T = std::__cxx11::stoi(argv[3]);
            }
            else
            {
                filename1 = "Files/GraphBlock.txt";
                filename2 = "Files/Edges.txt";
                filename3 = "Files/Output.txt";
                filename4 = "Files/Results.txt";

                d = 1;
                T = 1;
            }


            bool print_weights = false;

            graph_distributed *example_graph = new graph_distributed(d, filename1, filename2, world);

            int N = (*example_graph).num_vertices;
            int N_halo = (*example_graph).num_halo_vertices;

            world.log("Processor %i: Number of vertices is %i", s, N);
            world.log("Processor %i: Number of halo vertices is %i", s, N_halo);


            std::vector<Matrix*> weights;
            read_weights(d, weights, print_weights);


            gnn_distributed *example_gnn = new gnn_distributed(T, example_graph, weights);



            auto time = bulk::util::timer();

            example_gnn->forward_pass(world);

            auto msecs = time.get();


            world.sync();

            if(s==0)
            {
                world.log("Milliseconds passed: %lf", msecs);
                write_output_file(example_graph, filename3);
                write_result_file(d, T, p, msecs, filename4);
            }


        });



        return 0;
    }






