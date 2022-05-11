#include "graph_distributed.h"

graph_distributed::graph_distributed()
{

}

graph_distributed::graph_distributed(int d, std::string filename_vertices, std::string filename_edges, bulk::world& world)
{
    this->dimension = d;
    build_graph_vertices(filename_vertices, world);
    build_graph_edges(filename_edges);
    build_halo(filename_vertices);
}

graph_distributed::~graph_distributed()
{
    this->vertices.clear();
}


void graph_distributed::build_graph_vertices(std::string filename, bulk::world& world)
{
    std::string content;

    std::ifstream graph_file(filename.c_str(), std::ios::in);

    std::string substring;

    ColVector new_vertex = ColVector(this->dimension);
    int i;
    int local_index, global_index;
    int pid;
    int s = world.rank();

    if ( !graph_file.is_open() )
    {
        std::cout << "The file " << filename << " could not be opened." << "\n";
    }
    else
    {

        local_index=0;
        global_index=0;
        while(getline(graph_file, content, '\n'))
        {
            std::stringstream content_line(content);

            getline(content_line, substring, ' ');
            std::stringstream processor(substring);
            processor >> pid;

            if(pid == s)
            {
                i=0;
                while(getline(content_line, substring, ' '))
                {
                    std::stringstream coefficient(substring);
                    coefficient >> new_vertex[i];
                    i++;
                }

                this->vertices.push_back(new vertex(new_vertex));
                this->global_vertex_indices.push_back(global_index);
                this->local_vertex_indices.push_back(local_index);

                local_index++;
            }
            else
            {
                this->local_vertex_indices.push_back(-1);
            }

            global_index++;
        }
    }

    this->num_vertices = (this->vertices).size();

    graph_file.close();

}


void graph_distributed::build_graph_edges(std::string filename)
{
    std::string content;

    std::ifstream graph_file(filename.c_str(), std::ios::in);

    std::string substring;

    int i1, i2;
    int local_i1, local_i2;
    Edge halo_edge;


    if ( !graph_file.is_open() )
    {
        std::cout << "The file " << filename << " could not be opened." << "\n";
    }
    else
    {
        while(getline(graph_file, content, '\n'))
        {
            std::stringstream content_line(content);

            getline(content_line, substring, ' ');
            std::stringstream index1(substring);
            index1 >> i1;

            getline(content_line, substring, ' ');
            std::stringstream index2(substring);
            index2 >> i2;

            local_i1 = this->local_vertex_indices[i1];
            local_i2 = this->local_vertex_indices[i2];

            if(local_i1 > -1)
            {

                if(local_i2 > -1)
                {
                    (*(this->vertices)[local_i1]).neighbor_vertices.push_back((this->vertices)[local_i2]);
                    (*(this->vertices)[local_i2]).neighbor_vertices.push_back((this->vertices)[local_i1]);
                }
                else
                {
                    if(std::find(this->halo_vertices.begin(), this->halo_vertices.end(), i2) == this->halo_vertices.end())
                    {
                        this->halo_vertices.push_back(i2);
                    }

                    this->halo_edges.push_back(new Edge);
                    halo_edge(0) = i1;
                    halo_edge(1) = i2;
                    *(this->halo_edges.back()) = halo_edge;
                }

            }
            else
            {

                if(local_i2 > -1)
                {
                    if(std::find(this->halo_vertices.begin(), this->halo_vertices.end(), i1) == this->halo_vertices.end())
                    {
                        this->halo_vertices.push_back(i1);
                    }

                    this->halo_edges.push_back(new Edge);
                    halo_edge(0) = i2;
                    halo_edge(1) = i1;
                    *(this->halo_edges.back()) = halo_edge;
                }

            }

        }
    }

    graph_file.close();

}


void graph_distributed::build_halo(std::string filename)
{
    std::string content;

    std::ifstream graph_file(filename.c_str(), std::ios::in);

    std::string substring;

    ColVector new_vertex = ColVector(this->dimension);
    int i;
    int local_index, global_index;
    int pid;
    Edge edge;
    int local_i1, local_i2;
    int N_halo_edges = this->halo_edges.size();
    vertex *current_vertex;

    if ( !graph_file.is_open() )
    {
        std::cout << "The file " << filename << " could not be opened." << "\n";
    }
    else
    {
        local_index=this->num_vertices;
        global_index=0;


        while(getline(graph_file, content, '\n'))
        {
            if(std::find(this->halo_vertices.begin(), this->halo_vertices.end(), global_index) != this->halo_vertices.end())
            {
                std::stringstream content_line(content);

                getline(content_line, substring, ' ');
                std::stringstream processor(substring);
                processor >> pid;

                i=0;
                while(getline(content_line, substring, ' '))
                {
                    std::stringstream coefficient(substring);
                    coefficient >> new_vertex[i];
                    i++;
                }
                this->vertices.push_back(new vertex(new_vertex));
                this->global_vertex_indices.push_back(global_index);
                this->local_vertex_indices[global_index] = local_index;

                local_index++;

                (*(this->vertices.back())).set_owner(pid);
            }

            global_index++;
        }
    }

    this->num_halo_vertices = (this->vertices).size() - this->num_vertices;
    this->halo_vertices.clear();


    graph_file.close();


    for(i=0; i<N_halo_edges; i++)
    {
        edge = *(this->halo_edges.back());
        local_i1 = this->local_vertex_indices[edge(0)];
        local_i2 = this->local_vertex_indices[edge(1)];

        this->halo_edges.pop_back();

        (*(this->vertices)[local_i1]).neighbor_vertices.push_back((this->vertices)[local_i2]);
        (*(this->vertices)[local_i2]).neighbor_vertices.push_back((this->vertices)[local_i1]);


        current_vertex = (this->vertices)[local_i1];
        pid = ((this->vertices)[local_i2])->get_owner();

        if(std::find(this->requested_vertices.begin(), this->requested_vertices.end(), local_i1) == this->requested_vertices.end())
        {
            this->requested_vertices.push_back(local_i1);
        }
        if(std::find(((*current_vertex).neighbor_processors).begin(), ((*current_vertex).neighbor_processors).end(), pid) == ((*current_vertex).neighbor_processors).end())
        {
            ((*current_vertex).neighbor_processors).push_back(pid);
        }


    }


}






