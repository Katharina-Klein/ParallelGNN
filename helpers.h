
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>

    typedef Eigen::MatrixXf Matrix;
    typedef Eigen::RowVectorXf RowVector;
    typedef Eigen::VectorXf ColVector;


#include "graph_distributed.h"


#ifndef HELPERS_H
    #define HELPERS_H

ColVector sigmoid(ColVector input_vector);
ColVector hyptan(ColVector input_vector);

ColVector sigmoid_derivative(ColVector input_vector);
ColVector hyptan_derivative(ColVector input_vector);

Matrix vector_from_left(Matrix M, ColVector v);
Matrix vector_from_right(Matrix M, ColVector v);


Matrix read_weight_file(std::string filename);
void read_weights(int d, std::vector<Matrix*> &weights, bool print_weights);


void write_output_file(graph_distributed* graph_name, std::string filename);
void write_result_file(int d, int T, int p, double msecs, std::string filename);





#endif // HELPERS_H

