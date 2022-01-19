#include <vector>
#include <set>
#include <map>
#include <utility>
#include <iostream>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


using namespace std;

//this is an Euler characteristic curve, as a real valued function with values in integers. This is what we compute locally
// inline std::vector< std::pair< double , int > >
inline std::tuple< std::map< double , int > , int, int>
compute_local_EC(
const std::vector< std::vector< std::pair< unsigned , double > > >& considered_graph,
const int & max_dimension = -1,
const bool & dbg = false
)

{
    if ( dbg )
    {
        cerr << "Here is the considered graph: \n";
        for ( size_t i = 0 ; i != considered_graph.size() ; ++i )
        {
            cerr << "i : " << i << endl;
            for ( size_t j = 0 ; j != considered_graph[i].size() ; ++j )
            {
                cerr << "(" << considered_graph[i][j].first << "," << considered_graph[i][j].second << "), ";
            }
            cerr << endl;
        }
    }

    //this is the list of simplices in the current dimension. We will overwrite it everytime we move to a higher dimension.
    std::vector< std::vector< unsigned > > simplices_in_current_dimension;
    simplices_in_current_dimension.reserve( considered_graph.size() );
    //this is a list which tells us what are the filtration values of the simplices in the simplices_in_current_dimension vector. I keed it not in a structure to make the code faster.
    std::vector< double > filtration_of_those_simplices;
    filtration_of_those_simplices.reserve( considered_graph.size() );

    //the value of Euler characteristic will only change at some number of real numbers, equal to the lenth of edges. We will keep in in the structure of a map:
    std::map< double , int > ECC;
    ECC.insert( std::make_pair( 0 , 1 ) );

    //first we will fill in simplices_in_current_dimension vector with the edges from central element to its neighbors with higher id:
    for ( size_t i = 0 ; i != considered_graph[0].size() ; ++i )
    {
        std::vector< unsigned > this_simplex = { 0 , considered_graph[0][i].first };
        simplices_in_current_dimension.push_back( this_simplex );
        filtration_of_those_simplices.push_back( considered_graph[0][i].second );

        //changes to ECC due to those edges:
        if ( dbg )cerr << "######Change of ECC at the level : " << considered_graph[0][i].second  << " by: -1"  << endl;
        std::map< double , int >::iterator it = ECC.find( considered_graph[0][i].second );
        if ( it != ECC.end() )
        {
            double val = it->second;
            val += -1;
            it->second = val;
        }
        else
        {
            ECC.insert( std::make_pair( considered_graph[0][i].second , -1 ) );
        }
    }


    if ( dbg )
    {
        cerr << "simplices_in_current_dimension : \n";
        for ( size_t i = 0 ; i != simplices_in_current_dimension.size() ; ++i )
        {
            cerr << "[";
            for ( size_t j = 0 ; j != simplices_in_current_dimension[i].size() ; ++j )
            {
                cerr << simplices_in_current_dimension[i][j] << ",";
            }
            cerr << "] --> " << filtration_of_those_simplices[i] << endl;
        }
    }

    //We do not need to list the neighbors of the certal vertex, as they are vertices with numbers between 1 and considered_graph[0].size()
    //std::set< unsigned > neighs_of_central_vertex;
    //for ( size_t i = 0 ; i != considered_graph[0].size() ; ++i )
    //{
    //    neighs_of_central_vertex.insert( considered_graph[0][i].first );
    //    if ( dbg )cerr << "Neighs of the central vertex : " << considered_graph[0][i].first << endl;
    //}
    unsigned last_neigh_of_current_vertex = considered_graph[0].size();
    if ( dbg )cerr << "last_neigh_of_current_vertex : " << last_neigh_of_current_vertex << endl;


    //now we can compute all the neighbors of the created 1-dimensional simplices:
    std::vector< std::vector<unsigned> > common_neighs( simplices_in_current_dimension.size() );
    for ( size_t i = 0 ; i != simplices_in_current_dimension.size() ; ++i )
    {
        //we need to check which vertices among neighs_of_neighs_of_centre are common neighs of the vertices in this simplex.
        unsigned the_other_vertex = simplices_in_current_dimension[i][1];
        if ( dbg )cerr << "We will search for the common neighbors of the edge [0," <<  the_other_vertex << "] : \n";
        std::vector< unsigned > neighs;
        neighs.reserve( considered_graph[the_other_vertex].size() );
        if ( dbg )cerr << "The common neigh is : ";
        for ( size_t j = 0 ; j != considered_graph[the_other_vertex].size() ; ++j )
        {
            neighs.push_back( considered_graph[the_other_vertex][j].first );
            if ( dbg )cerr << considered_graph[the_other_vertex][j].first << " , ";
        }
        if ( dbg )cerr << endl;
        common_neighs[i] = neighs;
    }

    if ( dbg )
    {
        cerr << "Here is final common_neighs list. \n";
        for ( size_t i = 0 ; i != common_neighs.size() ; ++i )
        {
            cerr << "For the edge : [" << simplices_in_current_dimension[i][0] << "," << simplices_in_current_dimension[i][1] << "] we have the common neighs : ";
            for ( size_t j = 0 ; j != common_neighs[i].size() ; ++j )
            {
                cerr << common_neighs[i][j] << " ";
            }
            cerr << ". Moreover its filtration is : " << filtration_of_those_simplices[i];
            cerr << endl;
        }
    }

    //ToDo, compute the contributio to the Euler characteristic!!! Again, we can optymize it here by not grouping everytihing - each contribution will be at a value of a filtration of an edge - we can perhaps use this fact to accumulate it all in a smart way.


    //We will use this datastructure for quick computations of intersections of neighbor lists
    std::vector< std::set<unsigned> > neighs_of_vertices( considered_graph.size() );
    for ( size_t i = 0 ; i != considered_graph.size() ; ++i )
    {
        std::set< unsigned > neighs_of_this_graph;
        for ( size_t j = 0 ; j != considered_graph[i].size() ; ++j )
        {
            neighs_of_this_graph.insert( considered_graph[i][j].first );
        }
        neighs_of_vertices[i] = neighs_of_this_graph;
    }

    // keep track of the number of simplices discovered
    int number_of_simplices = 1 + simplices_in_current_dimension.size();

    //Now we have created the list of edges, and each of the edge is equipped with common_neighs and all filtration values. Now, we can now create all higher dimensional simplices:
    unsigned dimension = 2;
    int dimm = 1;
    while ( !simplices_in_current_dimension.empty() )
    {

        if ((dimension-1) == max_dimension) {
          break;
        }

        //first we declare all cointainters that we need.
        std::vector< std::vector< unsigned > > new_simplices_in_current_dimension;
        new_simplices_in_current_dimension.reserve( simplices_in_current_dimension.size() );
        std::vector< double > new_filtration_of_those_simplices;
        new_filtration_of_those_simplices.reserve( filtration_of_those_simplices.size() );
        std::vector< std::vector<unsigned> > new_common_neighs;
        new_common_neighs.reserve( simplices_in_current_dimension.size() );

        //the real computations begins here:
        for ( size_t i = 0 ; i != simplices_in_current_dimension.size() ; ++i )
        {
            if ( dbg )
            {
                cerr << "Consider simplex : [";
                for ( size_t aa = 0 ; aa != simplices_in_current_dimension[i].size() ; ++aa )cerr << simplices_in_current_dimension[i][aa] << ", ";
                cerr << "]. " << endl;
                cerr << "common_neighs[i].size() : " << common_neighs[i].size() << endl;
            }

            //let us check if we can extend simplices_in_current_dimension[i]
            for ( size_t j = 0 ; j !=  common_neighs[i].size() ; ++j )
            {
                if ( dbg )cerr << "It can be extended by adding a vertex : " << common_neighs[i][j] << endl;

                //we can extend simplices_in_current_dimension[i] by adding vertex common_neighs[i][j]
                std::vector< unsigned > new_simplex( dimension+1 );
                for ( size_t k = 0 ; k != dimension ; ++k )new_simplex[k] = simplices_in_current_dimension[i][k];
                new_simplex[dimension] = common_neighs[i][j];
                new_simplices_in_current_dimension.push_back( new_simplex );
                number_of_simplices++;
                if ( dbg )
                {
                    cerr << "Adding new simplex : ";
                    for ( size_t aa = 0 ; aa != new_simplex.size() ; ++aa )cerr << new_simplex[aa] << " ";
                    cerr << endl;
                }

                //now once we have the new simplex, we need to compute its filtration and common neighs
                //let us start with the filtration. We will set it up initially to the filtration of simplices_in_current_dimension[i]:
                double filtration_of_this_simplex = filtration_of_those_simplices[i];
                for ( size_t k = 0 ; k != simplices_in_current_dimension[i].size() ; ++k )
                {
                    //check what is the weight of an edge from simplices_in_current_dimension[i][k] to common_neighs[i][j]
                    double length_of_this_edge = 0;
                    for ( size_t l = 0 ; l != considered_graph[ simplices_in_current_dimension[i][k] ].size() ; ++l )
                    {
                        if ( considered_graph[ simplices_in_current_dimension[i][k] ][l].first == common_neighs[i][j] )
                        {
                            length_of_this_edge = considered_graph[ simplices_in_current_dimension[i][k] ][l].second;
                            //break;
                        }
                    }
                    if ( length_of_this_edge > filtration_of_this_simplex )filtration_of_this_simplex = length_of_this_edge;
                }
                new_filtration_of_those_simplices.push_back( filtration_of_this_simplex );

                if ( dbg )cerr << "#####Change of ECC at the level : " << filtration_of_this_simplex  << " by: " << dimm << endl;
                std::map< double , int >::iterator it = ECC.find( filtration_of_this_simplex );
                if ( it != ECC.end() )
                {
                    double val = it->second;
                    val += dimm;
                    it->second = val;
                }
                else
                {
                    ECC.insert( std::make_pair( filtration_of_this_simplex , dimm ) );
                }


                if ( dbg )cerr << "The filtration of this simplex is : " << filtration_of_this_simplex << endl;

                //now we still need to deal with the common neighbors.
                // complexity O(common_neighs[i].size() * log(neighs_of_vertices[new_vertex].size()))
                std::vector<unsigned> neighs_of_new_simplex;
                neighs_of_new_simplex.reserve( common_neighs[i].size() );
                unsigned new_vertex = common_neighs[i][j];
                for ( size_t k = 0 ; k != common_neighs[i].size() ; ++k )
                {
                    // there seems to be a mistake, there was no k in this loop
                    // I substituted j --> k
                    // if find() is not successful returns an iterator to set::end
                    // complexity of find() is log in size
                    if ( neighs_of_vertices[new_vertex].find( common_neighs[i][k] ) != neighs_of_vertices[new_vertex].end() )
                    {
                        neighs_of_new_simplex.push_back( common_neighs[i][k] );
                    }
                }
                new_common_neighs.push_back( neighs_of_new_simplex );
            }
        }

        if ( dbg )
        {
            cerr << "Moving to the next dimension, press enter to continue your journey. \n";
            getchar();
        }

        simplices_in_current_dimension = new_simplices_in_current_dimension;
        filtration_of_those_simplices = new_filtration_of_those_simplices;
        common_neighs = new_common_neighs;
        dimension++;
        dimm = dimm*(-1);
    }

    if ( dbg )cerr << "Out of the loop, return result. \n";



    // std::vector< std::pair< double , int > > result;
    // result.reserve( ECC.size() );
    // for ( map< double , int >::iterator it = ECC.begin() ; it != ECC.end() ; ++it )
    // {
    //     if ( dbg ){cerr << it->first << " --> " << it->second << endl;}
    //     result.push_back( std::make_pair( it->first , it->second ) );
    // }

    // return std::make_pair(ECC, number_of_simplices);
    return std::make_tuple(ECC, number_of_simplices, dimension-1);
}//compute_local_EC


PYBIND11_MODULE(_compute_local_EC_VR, m) {
    m.doc() = "compute_contributions_vertex test plugin"; // optional module docstring

    m.def("compute_contributions_vertex", &compute_local_EC, "compute contributions single vertex");
}
