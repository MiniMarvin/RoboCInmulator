/**
 * TODO: it's necessary to build the genomes analysis in the following way
 * The pool must be splited in the test pool and the training pool, it is, the test pool
 * contains the best quoted gens and the test pool contains the gens which we want to achive
 * a better performance. 
 * Once a generation overs and some gens achieved a performance better than the ones in the 
 * testing pool they must update the testing pool with the new gens. The ones which achieve
 * a better performance are going to substitute the ones with the worst performance. For allow
 * a better performance achievement and less enviesation, we must add to the training pool more
 * gens than the ones which are going to be tested randomly, something like twice the number of
 * gens in every generation, like that we promote a bigger variance of the genome itself and
 * there must exists a game against a stupid gen to verify if the system stopped learning how
 * to win against a stupid player.
 * 
 * 
 * Problems with the training technique: There may exists an enviesation of the way that the 
 * system plays once it cannot play against gens outside of the testing pool which may learn
 * to play in a really specific way.
 */
#ifndef __GENETIC_POOL
#define __GENETIC_POOL

// System includes
#include <iostream>
#include <math.h>
#include <time.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/lambda/bind.hpp>

// OpenNN includes
#include <opennn.h>


#define INF 0x3f3f3f3f

using namespace OpenNN;

typedef struct Gene {
	long id;
	double fitness;
	Vector<double> gen;
} Gene;

class geneticPool {
private:
	double mutationRate; // The mutation rate of the gens
	double crossoverRate; // The amount of every gen in the crossover process, 50% means equal distribution
	std::vector<Gene> train_pool; // The group of gens with the best fitness of all generations
	std::vector<Gene> test_pool; // The group of gens that are being tested in the generation
	Vector<int> structure; // The neural network topology

public:

	/**
	 * @brief      A pool for iterative learning with neural networks.
	 */
	geneticPool();
	
	/**
	 * @brief      A simple destructor instanciation.
	 */
	~geneticPool();

	/**
	 * @brief      Creates a mutation of a certain gen.
	 *
	 * @param[in]  gen   The gen to be mutated
	 *
	 * @return     The mutated version of the gen.
	 */
	Vector<double> mutate(Vector<double> gen);

	/**
	 * @brief      A procedure to make the crossover of two gens, putting parts of the gen B in gen A.
	 *
	 * @param[in]  a     The first gen.
	 * @param[in]  b     The second gen.
	 *
	 * @return     The crossover version of the gens.
	 */
	Vector<double> crossover(Vector<double> a, Vector<double> b);

	/**
	 * @brief      Gets a random gen in the test pool.
	 *
	 * @return     The gen.
	 */
	Gene get_testgen();

	/**
	 * @brief      Gets a specific gen in the test pool.
	 *
	 * @param[in]  num   The index of the gen.
	 *
	 * @return     The gen.
	 */
	Gene get_testgen(int num);

	/**
	 * @brief      Gets a specific gen in the train pool.
	 *
	 * @param[in]  num   The index of the gen.
	 *
	 * @return     The gen.
	 */
	Gene get_traingen(int num);

	/**
	 * @brief      Gets the train group.
	 *
	 * @return     The train group.
	 */
	std::vector<Gene> get_traingroup();

	/**
	 * @brief      Gets the test group.
	 *
	 * @return     The test group.
	 */
	std::vector<Gene> get_testgroup();

	/**
	 * @brief      Adds a gen to the train pool.
	 *
	 * @param[in]  gen   The params of the gen.
	 */
	void add_traingen(Vector<double>gen);

	/**
	 * @brief      Adds a gen to the test pool.
	 *
	 * @param[in]  gen   The params of the gen.
	 */
	void add_testgen(Vector<double>gen);

	/**
	 * @brief      Sets the train pool from params.
	 *
	 * @param[in]  genlist  The gen list.
	 */
	void set_traingen(std::vector< Vector<double> > genlist);

	/**
	 * @brief      Sets the test pool from params.
	 *
	 * @param[in]  genlist  The gen list.
	 */
	void set_testgen(std::vector< Vector<double> > genlist);

	/**
	 * @brief      Sets the train pool.
	 *             
	 * @param[in]  genlist  The gen list.
	 */
	void set_traingen(std::vector<Gene> genlist);

	/**
	 * @brief      Sets the test pool.
	 *             
	 * @param[in]  genlist  The gen list.
	 */
	void set_testgen(std::vector<Gene> genlist);

	/**
	 * @brief      Sets the fitness of a gen in the train pool.
	 *
	 * @param[in]  id       The identifier
	 * @param[in]  fitness  The fitness
	 */
	void set_trainfitness(int id, double fitness);

	/**
	 * @brief      Sets the fitness of a gen in the test pool.
	 *
	 * @param[in]  id       The identifier
	 * @param[in]  fitness  The fitness
	 */
	void set_testfitness(int id, double fitness);

	/**
	 * @brief      Build the next generation of genes.
	 */
	void next_gen();

	/**
	 * @brief      Loads all gens from a folder.
	 *
	 * @param[in]  folder         The folder
	 * @param[in]  network_model  The network model
	 */
	void load_folder(std::string folder, Vector<int> network_model);

	/**
	 * @brief      Saves all gens in a folder.
	 *
	 * @param[in]  folder         The folder
	 * @param[in]  network_model  The network model
	 */
	void save_folder(std::string folder, Vector<int> network_model);

	/**
	 * @brief      Loads all gens from a folder.
	 *
	 * @param[in]  folder         The folder
	 * @param[in]  network_model  The network model
	 */
	void load_folder(const char* folder, Vector<int> network_model);

	/**
	 * @brief      Saves all gens in a folder.
	 *
	 * @param[in]  folder         The folder
	 * @param[in]  network_model  The network model
	 */
	void save_folder(const char* folder, Vector<int> network_model);

	/**
	 * @brief      Loads all gens from a folder.
	 *
	 * @param[in]  folder         The folder
	 * @param[in]  network_model  The network model
	 */
	void load_folder(char* folder, Vector<int> network_model);

	/**
	 * @brief      Saves all gens in a folder.
	 *
	 * @param[in]  folder         The folder
	 * @param[in]  network_model  The network model
	 */
	void save_folder(char* folder, Vector<int> network_model);

	/**
	 * @brief      Sets the mutation rate.
	 *
	 * @param[in]  rate  The rate.
	 */
	void set_mutationrate(double rate);

	/**
	 * @brief      Sets the crossover rate.
	 *
	 * @param[in]  rate  The rate.
	 */
	void set_crossoverrate(double rate);

	/**
	 * @brief      Build a list of random gens.
	 *
	 * @param[in]  gen_num        The generate number
	 * @param[in]  network_model  The network model
	 *
	 * @return     The list of gens.
	 */
	std::vector<Gene> random_gens(int gen_num, Vector<int> network_model);


	// helper functions ---------------------------------------//
	
	/**
	 * @brief      Helper for sort the gens from the Gene Structure.
	 *
	 * @param[in]  a     The first gen.
	 * @param[in]  b     The second gen.
	 *
	 * @return     True if a > b and false otherwise.
	 */
	static bool sortbyfitness(const Gene &a, const Gene &b);

	/**
	 * @brief      Check if a certain folder exists.
	 *
	 * @param[in]  pathname  The pathname.
	 *
	 * @return     True if the folder exists and false otherwise.
	 */
	static bool exist_dir(std::string pathname);

	/**
	 * @brief      Creates a dir.
	 *
	 * @param[in]  pathname  The pathname
	 */
	static void create_dir(std::string pathname);

	/**
	 * @brief      Counts the number of files in a folder.
	 *
	 * @param[in]  pathname  The folder.
	 *
	 * @return     Number of files.
	 */
	static int count_files(std::string pathname);
};


#endif