#include "gnn_distributed.h"


gnn_distributed::gnn_distributed()
{

}

gnn_distributed::gnn_distributed(int T, graph_distributed *input_graph, std::vector<Matrix*> weights)
{
    this->dimension = (*input_graph).dimension;
    this->input_graph = input_graph;
    this->num_timesteps = T;
    this->weights = weights;
}

gnn_distributed::~gnn_distributed()
{
    this->weights.clear();
}


void gnn_distributed::forward_pass(bulk::world& world)
{

    int N = (this->input_graph)->num_vertices;
    int T = this->num_timesteps;
    int j;
    vertex *current_vertex;

    for( int i=1 ; i <= T ; i++)
    {

        this->communicate_states(world);

        for( j=0 ; j < N ; j++)
        {

            current_vertex = ((this->input_graph)->vertices)[j];
            (*current_vertex).message_passing(this->weights);

        }

        for( j=0 ; j < N ; j++)
        {

            current_vertex = ((this->input_graph)->vertices)[j];
            (*current_vertex).set_hidden_state();

        }

    }
}



void gnn_distributed::communicate_states(bulk::world& world)
{
    auto message_queue = bulk::queue<int, float[]>(world);

    graph_distributed *graph = this->input_graph;
    int start = graph->num_vertices;
    int halo = graph->num_halo_vertices;
    int n_requested_vertices = (graph->requested_vertices).size();
    int n_procs;
    int i,j,k;
    vertex *halo_vertex;
    int local_vertex_id, global_vertex_id, pid;
    ColVector hidden_state;
    std::vector<float> state_values;
    for(j=0; j<this->dimension; j++)
    {
        state_values.push_back(0);
    }

    for(i = 0; i < n_requested_vertices; i++)
    {
        local_vertex_id = (graph->requested_vertices)[i];
        global_vertex_id = (graph->global_vertex_indices)[local_vertex_id];

        halo_vertex = (graph->vertices)[local_vertex_id];


        hidden_state = (*halo_vertex).get_hidden_state();

        for(j=0; j<this->dimension; j++)
        {
            state_values[j] = hidden_state[j];
        }

        n_procs = ((*halo_vertex).neighbor_processors).size();

        for(k=0; k<n_procs; k++)
        {
            pid = (*halo_vertex).neighbor_processors[k];
            message_queue(pid).send(global_vertex_id, state_values);
        }

    }

    world.sync();

    for (auto [index, values] : message_queue)
    {
        local_vertex_id = (graph->local_vertex_indices)[index];

        for(j=0; j<this->dimension; j++)
        {
            hidden_state[j] = values[j];
        }


        halo_vertex = (graph->vertices)[local_vertex_id];
        (*halo_vertex).initialize_state(hidden_state);

    }


}




