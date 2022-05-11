#include <vector>
#include <Eigen/Core>

    typedef Eigen::MatrixXf Matrix;
    typedef Eigen::RowVectorXf RowVector;
    typedef Eigen::VectorXf ColVector;

#ifndef VERTEX_H
    #define VERTEX_H
    class vertex
    {

    public:
        vertex();
        vertex(ColVector initial_state);
        ~vertex();


        std::vector<vertex*> neighbor_vertices;
        std::vector<int> neighbor_processors;


        void initialize_state(ColVector initial_state);
        ColVector get_hidden_state();
        ColVector get_hidden_state_temp();

        int get_owner();
        void set_owner(int pid);

        float readout_function();

        void message_passing(std::vector<Matrix*> weights);
        void set_hidden_state();


    private:

        ColVector hidden_state;
        ColVector hidden_state_temp;
        ColVector initial_hidden_state;

        int processor;

        ColVector gather_messages();
        void update_state(ColVector neighbor_states, std::vector<Matrix*> weights);


    protected:




    };


#endif // VERTEX_H

