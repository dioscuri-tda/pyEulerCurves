#include "Bitmap_cubical_complex.h"
#include "Bitmap_cubical_complex_periodic_boundary_conditions.h"

#include <cstdlib>
#include <chrono>
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace std::chrono;


//the main engine to do the computations
template< typename cub_cmplx , typename T>
std::vector< std::pair<T,int> > compute_local_ECC( cub_cmplx& b , double fraction_of_initial_elements_to_consider )
{
	bool dbg = false;

	//std::cerr << "fraction_of_initial_elements_to_consider : " << fraction_of_initial_elements_to_consider << std::endl;

	//we assume here that we have two layers of top dimensional cells. That gives five slices in this direction.
    //In total we have b.size() elements. We need filtration and dimension of the initial 2/5 of them.
    unsigned lower_bound = (unsigned)( 2*fraction_of_initial_elements_to_consider*b.size() );
    unsigned upper_bound = (unsigned)( 4*fraction_of_initial_elements_to_consider*b.size() );
    if ( dbg )
    {
		std::cerr << "b.size() : " << b.size() << std::endl;
		std::cerr << "lower_bound : " << lower_bound << std::endl;
		std::cerr << "upper_bound : " << upper_bound << std::endl;
		getchar();
	}


    std::vector< std::pair<double,int> > result;
    result.reserve( upper_bound -lower_bound  );
    for ( size_t i = lower_bound ; i != upper_bound ; ++i )
    {
		 int expp = 1;
         if ( b.get_dimension_of_a_cell(i)%2 == 1 )expp = -1;

		 if ( dbg )
		 {
			std::cerr << "Adding contribution : " << b.get_cell_data(i) << " with the coef : " << expp << " and dimension : " << b.get_dimension_of_a_cell(i) << std::endl;
			getchar();
		}

         std::pair< double, int > p = std::make_pair( b.get_cell_data(i) , expp );
         result.push_back( p );
    }

    //now sort result according to the first coordinate:
    std::sort( result.begin() , result.end() );

    if ( dbg )
    {
		std::cerr << "Here is the result: \n";
		for ( size_t i = 0 ; i != result.size() ; ++i )
		{
			std::cerr << result[i].first << " , " << result[i].second << std::endl;
		}
		std::cerr << "End.\n";
	}

	//now, once it is sorted, let us compress the result. Firstly, check how many different values we have in result.first;
	int number_of_different_values = 0;
	for ( size_t i = 1 ; i != result.size() ; ++i )
	{
		if ( result[i].first != result[i-1].first )++number_of_different_values;
	}
	++number_of_different_values;//we do not count the last one
	if ( dbg )std::cerr << "number_of_different_values : " << number_of_different_values << std::endl;

    //now we can sparsify the result:
    std::vector< std::pair<T,int> > sparsified_result;
    sparsified_result.reserve( number_of_different_values );
    int sum = result[0].second;
    for ( size_t i = 1 ; i != result.size() ; ++i )
    {
		if ( result[i].first != result[i-1].first )
		{
			sparsified_result.push_back( std::make_pair( result[i-1].first , sum ) );
			sum = 0;
		}
		sum += result[i].second;
	}
	//and we still need to add the last value:
	sparsified_result.push_back( std::make_pair( result[ result.size()-1 ].first , sum ) );

    return sparsified_result;
}//compute_local_ECC


/**
 * function to interface in non-periodic case
 * This function take two filtration on maximal cells in two constitutive slices of a cubical complex. It costruct it, and compute Euler Characteristic Curve (ECC) of the
 * first layer, withouth the upper part. See a picture in the publication.
 * This function works when no periodic boundary conditions are to be applied.
 **/
template <typename T>
std::vector< std::pair<T,int> > compute_local_euler_from_two_constitutive_slices_no_periodic_b_cond( const std::vector< T >& data , const std::vector< unsigned >& sizes )
{
    // auto start = high_resolution_clock::now();
    Bitmap_cubical_complex_base<T> b(sizes, data);
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<seconds>(stop - start);
    // std::cout << "Creation of a bitmap : " << duration.count() << std::endl;
		//
    // std::cerr << "b.size() : " << b.size() << std::endl;

    //in non periodic case every complex that have two layers of cubes in its last two dimensions will have 5 layers in the whole bitmap (draw 2d example to see it)
    double fraction_of_initial_elements_to_consider = 1./5.;
    // start = high_resolution_clock::now();
    std::vector< std::pair<T,int> > lecc = compute_local_ECC< Bitmap_cubical_complex_base<T> , T >( b , fraction_of_initial_elements_to_consider );
    // stop = high_resolution_clock::now();
    // duration = duration_cast<seconds>(stop - start);
    // std::cout << "Computaion of ECC : " << duration.count() << std::endl;
    return lecc;
}//compute_local_euler_from_two_constitutive_slices


/**
 * function to interface in periodic case
 * This function take two filtration on maximal cells in two constitutive slices of a cubical complex. It costruct it, and compute Euler Characteristic Curve (ECC) of the
 * first layer, withouth the upper part. See a picture in the publication.
 * This function works for periodic boundary conditions are to be applied.
 * NOTE: to make it work propertly all but the last element in const std::vector< bool >& dimensions_of_periodicity should be set to TRUE!!!!
 **/
template <typename T>
std::vector< std::pair<T,int> > compute_local_euler_from_two_constitutive_slices_periodic_b_cond( const std::vector< T >& data , const std::vector< unsigned >& sizes , const std::vector< bool >& dimensions_of_periodicity )
{
	//be careful for periodic case!!! We should have periodic direction everywhere except from the direction where complex has thickness 2.
		// auto start = high_resolution_clock::now();
    Bitmap_cubical_complex_periodic_boundary_conditions_base<T> b(sizes, data, dimensions_of_periodicity);
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<seconds>(stop - start);
    // std::cout << "Creation of a bitmap : " << duration.count() << std::endl;

	//in periodic case every complex that have two layers of cubes in its last two dimensions will have 4 layers in the whole bitmap (draw 2d example to see it)
		double fraction_of_initial_elements_to_consider = 1./4.;
		// start = high_resolution_clock::now();
		std::vector< std::pair<T,int> > ecc = compute_local_ECC< Bitmap_cubical_complex_periodic_boundary_conditions_base<T> , T >( b , fraction_of_initial_elements_to_consider );
		// stop = high_resolution_clock::now();
    // duration = duration_cast<seconds>(stop - start);
    // std::cout << "Computaion of ECC : " << duration.count() << std::endl;
	return ecc;
}//compute_local_euler_from_two_constitutive_slices_periodic_b_cond


template <typename T>
void print_output( const std::vector< std::pair<T,int> >& ecc )
{
std::cerr << "Here is the output : \n";
    for ( size_t i = 0 ; i != ecc.size() ; ++i )
    {
		std::cout << ecc[i].first << " " << ecc[i].second << std::endl;
	}
}


template <typename T>
void store_output( const std::vector< std::pair<T,int> >& ecc , const char* filename )
{
	std::ofstream out;
	out.open( filename );
    for ( size_t i = 0 ; i != ecc.size() ; ++i )
    {
		out << ecc[i].first << " " << ecc[i].second << std::endl;
	}
	out.close();
}


// int main( int argc, char** argv )
// {
//     // 13 14 15 16
//     // 9  10 11 12
//     // 5  6  7  8
//     // 1  2  3  4
//     //std::vector< double > data = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
//
//     //0 0 0 0
//     //0 1 1 0
//     //0 1 1 0
//     //0 0 0 0
//     //std::vector< double > data1 = {0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0};
//     std::vector< float > data0 = {100,100,100,3,3,3};
//     std::vector< float > data1 = {3,3,3,1,5,1};
//     std::vector< float > data2 = {1,5,1,1,1,1};
//     std::vector< float > data3 = {1,1,1,100,100,100};
//
//
//
//     std::vector< unsigned > dimensions = {3,2};
//
//     std::vector< bool > dimensions_of_periodicity = {true,true};
//     std::vector< std::pair<float,int> > cons = compute_local_euler_from_two_constitutive_slices_no_periodic_b_cond( data1 , dimensions );
//     std::vector< std::pair<float,int> > ecc = compute_local_euler_from_two_constitutive_slices_periodic_b_cond<float>( data1 , dimensions , dimensions_of_periodicity );
//
//
//     std::vector< std::pair<float,int> > ecc0 = compute_local_euler_from_two_constitutive_slices_no_periodic_b_cond<float>( data0 , dimensions );
//     print_output( ecc0 );
//
//     std::vector< std::pair<float,int> > ecc1 = compute_local_euler_from_two_constitutive_slices_no_periodic_b_cond<float>( data1 , dimensions );
//     print_output( ecc1 );
//
//     std::vector< std::pair<float,int> > ecc2 = compute_local_euler_from_two_constitutive_slices_no_periodic_b_cond<float>( data2 , dimensions );
//     print_output( ecc2 );
//
//     std::vector< std::pair<float,int> > ecc3 = compute_local_euler_from_two_constitutive_slices_no_periodic_b_cond<float>( data3 , dimensions );
//     print_output<float>( ecc3 );
//
//
//     // srand( 1 );//fix a random number generator, so that we will always get the same sequence of numbers
//     // unsigned n = 5000;
//     // //now we will generate a random bitmap of a size n by n by 2:
//     // std::vector< double > data(n*n*2);
//     // for ( size_t i = 0 ; i != n*n*2 ; ++i )
//     // {
//     //     data[i] = rand()%255; // (double)RAND_MAX;
//     // }
//     // std::vector< unsigned > dimensions = {n,n,2};
//     // const std::vector< bool > dimensions_of_periodicity = {true, true, false};
//     // //without periodic b. cond
//     // std::vector< std::pair<double,int> > ecc = compute_local_euler_from_two_constitutive_slices_no_periodic_b_cond<double>( data , dimensions );
//
//     //with periodic b. cond
//     //std::vector< std::pair<double,int> > ecc = compute_local_euler_from_two_constitutive_slices_periodic_b_cond<double>( data , dimensions , dimensions_of_periodicity );
//
//     //store_output( ecc , "output" );
//
//     return 0;
// }



std::vector< std::pair<double,int> >
specialization_double_compute_local_euler_from_two_constitutive_slices_no_periodic_b_cond( const std::vector< double >& data ,
	                                                                                      const std::vector< unsigned >& sizes)
{
	return compute_local_euler_from_two_constitutive_slices_no_periodic_b_cond( data , sizes);
}


PYBIND11_MODULE(_compute_local_EC_cubical, m) {
    m.doc() = "test plugin"; // optional module docstring

    m.def("compute_contributions_two_slices",
		      &specialization_double_compute_local_euler_from_two_constitutive_slices_no_periodic_b_cond,
					"specialization_double_compute_local_euler_from_two_constitutive_slices_no_periodic_b_cond");
}
