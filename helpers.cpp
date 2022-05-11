#include "helpers.h"

ColVector sigmoid(ColVector input_vector)
{
    int numrows = input_vector.rows();
    ColVector output_vector(numrows);

    for(int i=0 ; i < input_vector.rows() ; i++)
    {
        output_vector[i] = 1 / (1 + exp(input_vector[i]));
    }

    return output_vector;
}


ColVector hyptan(ColVector input_vector)
{
    int numrows = input_vector.rows();
    ColVector output_vector(numrows);

    for(int i=0 ; i < input_vector.rows() ; i++)
    {
        output_vector[i] = tanh(input_vector[i]);
    }

    return output_vector;
}


ColVector sigmoid_derivative(ColVector input_vector)
{
    int numrows = input_vector.rows();
    float sigmoid_value;
    ColVector output_vector(numrows);

    for(int i=0 ; i < input_vector.rows() ; i++)
    {
        sigmoid_value = 1 / (1 + exp(input_vector[i]));
        output_vector[i] = sigmoid_value * (1 - sigmoid_value);
    }

    return output_vector;
}


ColVector hyptan_derivative(ColVector input_vector)
{
    int numrows = input_vector.rows();
    ColVector output_vector(numrows);

    for(int i=0 ; i < input_vector.rows() ; i++)
    {
        output_vector[i] = 1 - (tanh(input_vector[i]) * tanh(input_vector[i]));
    }

    return output_vector;
}



Matrix vector_from_left(Matrix M, ColVector v)
{
    Matrix output_matrix = M;
    int N = v.rows();
    int i,j;

    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            output_matrix(i,j) = M(i,j)*v(i);
        }
    }

    return output_matrix;

}



Matrix vector_from_right(Matrix M, ColVector v)
{
    Matrix output_matrix = M;
    int N = v.rows();
    int i,j;

    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            output_matrix(j,i) = M(j,i)*v(i);
        }
    }

    return output_matrix;

}



Matrix read_weight_file(int d1, int d2, std::string filename)
{
    std::string content;

    std::ifstream weight_file(filename.c_str(), std::ios::in);

    std::string substring;

    Matrix weights = Matrix(d1,d2);

    int i;
    int j=0;

    if ( !weight_file.is_open() )
    {
        std::cout << "The file " << filename << " could not be opened." << "\n";
    }
    else
    {
        while(getline(weight_file, content, '\n'))
        {
            std::stringstream content_line(content);
            i=0;
            while(getline(content_line, substring, ' '))
            {
                std::stringstream coefficient(substring);
                coefficient >> weights(j,i);
                i++;
            }
            j++;
        }
    }

    return weights;

}




void read_weights(int d, std::vector<Matrix*> &weights, bool print_weights)
{

    std::string filename = "Files/WeightsB.txt";
    weights.push_back(new Matrix(d,d));
    *(weights.back()) = read_weight_file(d, d, filename);

    filename = "Files/WeightsbVector.txt";
    weights.push_back(new Matrix(d,1));
    *(weights.back()) = read_weight_file(d, 1, filename);

    filename = "Files/WeightsWz.txt";
    weights.push_back(new Matrix(d,d));
    *(weights.back()) = read_weight_file(d, d, filename);

    filename = "Files/WeightsUz.txt";
    weights.push_back(new Matrix(d,d));
    *(weights.back()) = read_weight_file(d, d, filename);

    filename = "Files/WeightsWr.txt";
    weights.push_back(new Matrix(d,d));
    *(weights.back()) = read_weight_file(d, d, filename);

    filename = "Files/WeightsUr.txt";
    weights.push_back(new Matrix(d,d));
    *(weights.back()) = read_weight_file(d, d, filename);

    filename = "Files/WeightsW.txt";
    weights.push_back(new Matrix(d,d));
    *(weights.back()) = read_weight_file(d, d, filename);

    filename = "Files/WeightsU.txt";
    weights.push_back(new Matrix(d,d));
    *(weights.back()) = read_weight_file(d, d, filename);


    if(print_weights)
    {
        std::cout << "Matrix B:\n" << *(weights[0]) << std::endl;
        std::cout << "Matrix b:\n" << *(weights[1]) << std::endl;
        std::cout << "Matrix Wz:\n" << *(weights[2]) << std::endl;
        std::cout << "Matrix Uz:\n" << *(weights[3]) << std::endl;
        std::cout << "Matrix Wr:\n" << *(weights[4]) << std::endl;
        std::cout << "Matrix Ur:\n" << *(weights[5]) << std::endl;
        std::cout << "Matrix W:\n" << *(weights[6]) << std::endl;
        std::cout << "Matrix U:\n" << *(weights[7]) << std::endl;
    }


}





void write_output_file(graph_distributed* graph_name, std::string filename)
{

    std::ofstream output_file(filename.c_str(), std::ios::out);

    int i, global_index;
    int N = (graph_name->vertices).size();
    float output;

    if ( !output_file.is_open() )
    {
        std::cout << "The file " << filename << " could not be opened." << "\n";
    }
    else
    {
        for(i = 0; i < N; i++)
        {
            global_index = (graph_name->global_vertex_indices)[i];
            output = (*((graph_name->vertices)[i])).readout_function();
            output_file << global_index << " " << output << "\n";
        }
    }

    output_file.close();

}



void write_result_file(int d, int T, int p, double msecs, std::string filename)
{

    std::ofstream result_file(filename.c_str(), std::ios::app);


    if ( !result_file.is_open() )
    {
        std::cout << "The file " << filename << " could not be opened." << "\n";
    }
    else
    {
        result_file << "d = " << d << ", T = " << T << ", P = " << p << "\n";
        result_file << "Time in milliseconds = " << msecs << "\n";
    }

    result_file.close();

}





