#include "vertex.h"
#include "helpers.h"

vertex::vertex()
{

}

vertex::vertex(ColVector initial_state)
{
    this->hidden_state = initial_state;
    this->hidden_state_temp = initial_state;
    this->initial_hidden_state = initial_state;
}

vertex::~vertex()
{
    int rows = this->hidden_state.rows();
    this->hidden_state = ColVector::Zero(rows);
    this->hidden_state_temp = ColVector::Zero(rows);

    this->neighbor_vertices.clear();
}


void vertex::initialize_state(ColVector initial_state)
{
    this->hidden_state = initial_state;
    this->hidden_state_temp = initial_state;
}

ColVector vertex::get_hidden_state()
{
    return this->hidden_state;
}

ColVector vertex::get_hidden_state_temp()
{
    return this->hidden_state_temp;
}

int vertex::get_owner()
{
    return this->processor;
}

void vertex::set_owner(int pid)
{
    this->processor = pid;
}


// Readout function can be adjusted
float vertex::readout_function()
{
    return (this->hidden_state).mean();
}



void vertex::update_state(ColVector neighbor_states, std::vector<Matrix*> weights)
{

    int N_v = neighbor_states.size();

    if(N_v >= 1)
    {

        ColVector a_v = neighbor_states;


        ColVector helper_1;
        ColVector helper_2;
        ColVector helper_3;

        ColVector z_v;
        ColVector r_v;
        ColVector hidden_candidate;

        helper_1.noalias() = (*weights[0]) * a_v;
        a_v = helper_1.array() + (*weights[1]).array();

        helper_1.noalias() = (*weights[2]) * a_v;
        helper_2.noalias() = (*weights[3]) * this->hidden_state;
        helper_3 = helper_1.array() + helper_2.array();
        z_v = sigmoid( helper_3 );

        helper_1.noalias() = (*weights[4]) * a_v;
        helper_2.noalias() = (*weights[5]) * this->hidden_state;
        helper_3 = helper_1.array() + helper_2.array();
        r_v = sigmoid( helper_3 );

        helper_1.noalias() = (*weights[6]) * a_v;
        helper_2 = r_v.array() * this->hidden_state.array();
        helper_2 = (*weights[7]) * helper_2;
        helper_3 = helper_1.array() + helper_2.array();
        hidden_candidate = hyptan( helper_3 );

        helper_1 = (1 - z_v.array()) * this->hidden_state.array() + z_v.array() * hidden_candidate.array();

        this->hidden_state_temp = helper_1;

    }

}


ColVector vertex::gather_messages()
{

    int rows = this->hidden_state.rows();
    ColVector neighbor_states_sum = ColVector::Zero(rows);
    int N_v = (this->neighbor_vertices).size();

    for (int i=0 ; i < N_v ; i++)
    {
        neighbor_states_sum = neighbor_states_sum + (*(this->neighbor_vertices)[i]).get_hidden_state();
    }

    return neighbor_states_sum;

}



void vertex::message_passing(std::vector<Matrix*> weights)
{

    ColVector neighbor_states = this->gather_messages();

    this->update_state(neighbor_states, weights);


}



void vertex::set_hidden_state()
{
    this->hidden_state = this->hidden_state_temp;
}




