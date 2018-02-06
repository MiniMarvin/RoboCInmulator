// System includes
#include <iostream>
#include <math.h>
#include <time.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <boost/lambda/bind.hpp>

// OpenNN includes
#include <opennn.h>
#include "genetic_pool.hpp"



geneticPool::geneticPool() {
	srand(time(NULL));
	this->mutationRate = 0.05;
	this->crossoverRate = 0.05;
}

geneticPool::~geneticPool() {

}

Vector<double> geneticPool::mutate(Vector<double> gen) {
	for(int i = 0; i < gen.size(); i++) {
		double val = 1.0*rand()/RAND_MAX;
		
		if(val < this->mutationRate) {
			gen[i] += 2.0*rand()/RAND_MAX - 1.0;
		}
	}

	return gen;
}

Vector<double> geneticPool::crossover(Vector<double> a, Vector<double> b) {
	for(int i = 0; i < a.size(); i++) {
		double val = 1.0*rand()/RAND_MAX;
		
		if(val < this->crossoverRate) {
			a[i] = b[i];
		}
	}

	return a;
}

Gene geneticPool::get_testgen() {
	int num = rand()%this->test_pool.size();
	return test_pool[num];
}

Gene geneticPool::get_testgen(int num) {
	return test_pool[num];
}

Gene geneticPool::get_traingen(int num) {
	return train_pool[num];
}

std::vector<Gene> geneticPool::get_traingroup() {
	return this->train_pool;
}

std::vector<Gene> geneticPool::get_testgroup() {
	return this->test_pool;
}

void geneticPool::add_traingen(Vector<double> gen) {
	Gene buff;
	buff.id = train_pool.size();
	buff.fitness = 0;
	buff.gen = gen;
	this->train_pool.push_back(buff);
}

void geneticPool::add_testgen(Vector<double> gen) {
	Gene buff;
	buff.id = train_pool.size();
	buff.fitness = 0;
	buff.gen = gen;
	this->test_pool.push_back(buff);
}

void geneticPool::set_traingen(std::vector< Vector<double> > genlist) {
	std::vector<Gene> lst;
	Gene buff;

	for(int i = 0;i < genlist.size(); i++) {
		buff.gen = genlist[i];
		lst.push_back(buff);
	}
	this->train_pool = lst;
}

void geneticPool::set_testgen(std::vector< Vector<double> > genlist) {
	std::vector<Gene> lst;
	Gene buff;

	for(int i = 0;i < genlist.size(); i++) {
		buff.gen = genlist[i];
		lst.push_back(buff);
	}
	this->test_pool = lst;
}

void geneticPool::set_traingen(std::vector<Gene> genlist) {
	this->train_pool = genlist;
}

void geneticPool::set_testgen(std::vector<Gene> genlist) {
	this->test_pool = genlist;
}

void geneticPool::set_trainfitness(int id, double fitness) {
	this->train_pool[id].fitness = fitness;
}

void geneticPool::set_testfitness(int id, double fitness) {
	this->test_pool[id].fitness = fitness;
}

void geneticPool::next_gen() {
	std::vector<Gene> new_generation;
	std::vector<Gene> new_tests;
	Vector<double> buffgene;
	Gene buff;
	std::vector<Gene> old_tests = this->test_pool;
	std::vector<Gene> old_train = this->train_pool;

	buff.fitness = -INF;

	// order the lists by the fitness
	std::sort(old_train.begin(), old_train.end(), this->sortbyfitness);
	std::sort(old_tests.begin(), old_tests.end(), this->sortbyfitness);
	int gen_size = this->train_pool.size();

	for(int i = 0; i < gen_size/2; i++) {
		buff.gen = old_train[i].gen;
		buff.id = i;
		buff.fitness = old_train[i].fitness;
		new_generation.push_back(buff);
		for(int j = 0; j < gen_size; j++) {
			if(buff.gen == this->train_pool[j].gen) {
				std::cout << "\nGEN USED: " << j << "\n\n";
			}
		}
	}

	for (int i = 0; i < gen_size/2; i++) {
		int val = rand()%(gen_size/2);
		
		buff.gen = this->crossover(old_train[i].gen, old_train[val].gen);
		buff.gen = this->mutate(buff.gen);
		buff.id = gen_size/2 + i;
		buff.fitness = -INF;
		new_generation.push_back(buff);
	}

	// make the old generation become substituted by the trainned one
	int j = 0, k = 0;

	for(int i = 0; i < gen_size; i++) {
		if(old_tests[j].fitness > old_train[k].fitness) {
			new_tests.push_back(old_tests[j]);
			j++;
		}
		else {
			new_tests.push_back(old_train[k]);
			k++;
		}
	}

	for(int i = 0; i < this->test_pool.size(); i++) {
		std::cout << this->test_pool[i].fitness << " ";
	}
	std::cout << "\n";

	for(int i = 0; i < this->train_pool.size(); i++) {
		std::cout << this->train_pool[i].fitness << " ";
	}
	std::cout << "\n\n";	

	this->test_pool = new_tests;
	this->train_pool = new_generation;
}

bool geneticPool::exist_dir(std::string pathname) {
	struct stat info;

	if( stat( pathname.c_str(), &info ) != 0 ){
	    // printf( "cannot access %s\n", pathname );
	    return 0;
	}
	else if( info.st_mode & S_IFDIR ) {  // S_ISDIR() doesn't exist on my windows 
	    // printf( "%s is a directory\n", pathname );
	    return 1;
	}
	else {
	    // printf( "%s is no directory\n", pathname );
	    return 0;
	}
}

void geneticPool::create_dir(std::string pathname) {
	std::string mk = "mkdir " + pathname;
	system(mk.c_str());
}

void geneticPool::save_folder(std::string folder, Vector<int> network_model) {

	std::string train_path = folder + "/train/";
	std::string test_path = folder + "/test/";

	if(!this->exist_dir(folder)) {
		this->create_dir(folder);
	}

	if(!this->exist_dir(train_path)) {
		this->create_dir(train_path);
	}

	if(!this->exist_dir(test_path)) {
		this->create_dir(test_path);
	}

	MultilayerPerceptron network(network_model);

	for(int i = 0; i < this->train_pool.size(); i++) {
		network.set_parameters(this->train_pool[i].gen);
		std::string path = train_path + std::to_string(i) + ".xml";

		tinyxml2::XMLDocument* doc = network.to_XML();
    	tinyxml2::XMLError eResult = doc->SaveFile(path.c_str());
	}

	for(int i = 0; i < this->test_pool.size(); i++) {
		network.set_parameters(this->test_pool[i].gen);
		std::string path = test_path + std::to_string(i) + ".xml";

		tinyxml2::XMLDocument* doc = network.to_XML();
    	tinyxml2::XMLError eResult = doc->SaveFile(path.c_str());
	}
}

void geneticPool::load_folder(std::string folder, Vector<int> network_model) {

	std::string train_path = folder + "/train/";
	std::string test_path = folder + "/test/";

	if(!this->exist_dir(folder)) {
		throw(1);
	}

	if(!this->exist_dir(train_path)) {
		throw(2);
	}

	if(!this->exist_dir(test_path)) {
		throw(3);
	}

	MultilayerPerceptron network(network_model);
	Gene buff_gen;

	int train_num = this->count_files(train_path);
	int test_num = this->count_files(test_path);

	for(int i = 0; i < train_num; i++) {
		std::string path = train_path + std::to_string(i) + ".xml";

		tinyxml2::XMLDocument n_doc;
    	n_doc.LoadFile(path.c_str());
    	network.from_XML(n_doc);

    	buff_gen.fitness = -INF;
    	buff_gen.gen = network.arrange_parameters();
    	this->train_pool.push_back(buff_gen);
	}

	for(int i = 0; i < test_num; i++) {
		std::string path = test_path + std::to_string(i) + ".xml";

		tinyxml2::XMLDocument n_doc;
    	n_doc.LoadFile(path.c_str());
    	network.from_XML(n_doc);

    	buff_gen.fitness = -INF;
    	buff_gen.gen = network.arrange_parameters();
    	this->test_pool.push_back(buff_gen);
	}
}

void geneticPool::load_folder(const char* folder, Vector<int> network_model) {
	std::string pathname(folder);
	this->load_folder(pathname, network_model);
}

void geneticPool::save_folder(const char* folder, Vector<int> network_model) {
	std::string pathname(folder);
	this->save_folder(pathname, network_model);
}

void geneticPool::load_folder(char* folder, Vector<int> network_model) {
	std::string pathname(folder);
	this->load_folder(pathname, network_model);
}

void geneticPool::save_folder(char* folder, Vector<int> network_model) {
	std::string pathname(folder);
	this->save_folder(pathname, network_model);
}

void geneticPool::set_mutationrate(double rate) {
	this->mutationRate = rate;
}

void geneticPool::set_crossoverrate(double rate) {
	this->crossoverRate = rate;
}

bool geneticPool::sortbyfitness(const Gene &a, const Gene &b) {
	return (a.fitness > b.fitness); // order descending
}

std::vector<Gene> geneticPool::random_gens(int gen_num, Vector<int> network_model) {
	MultilayerPerceptron network;
	Vector<double> gen;
	std::vector<Gene> lst;
	Gene buff;

	for(int i = 0; i < gen_num; i++) {
		network.set(network_model);
		gen = network.arrange_parameters();

		buff.gen = gen;
		buff.fitness = -INF;
		buff.id = 0;
		lst.push_back(buff);
	}

	return lst;
}


int geneticPool::count_files(std::string pathname) {
  using namespace boost::filesystem;
  using namespace boost::lambda;

  path the_path(pathname);

  int cnt = std::count_if(
      directory_iterator(the_path),
      directory_iterator(),
      static_cast<bool(*)(const path&)>(is_regular_file) );

  return cnt;
}