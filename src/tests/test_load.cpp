// System includes
#include <iostream>
#include <math.h>
#include <time.h>
#include <cstdlib>
#include <string>
#include <exception>

// OpenNN includes
#include <opennn.h>
#include "../libs/genetic_pool.hpp"

using namespace OpenNN;

double f(double x) {
    // return 2*x - 5;
    return x*x -0.5*x;
}

double loss(double x, double y) {
    double y_ = f(x);
    return -((y - y_)*(y - y_));
}

int main(void) {

    srand(time(NULL));

    int nums[] = {1, 10, 10, 10, 1};
    Vector<int> structure(5, 0);
    
    for(int i = 0; i < 5; i++) {
        structure[i] = nums[i];
    }

    Vector<double> inputs(1, 0.0);
    Vector<double> outputs(1, 0.0);

    MultilayerPerceptron network(structure);

    geneticPool pool;

    std::vector<Gene> test_gens = pool.random_gens(10, structure);
    std::vector<Gene> train_gens = test_gens;
    try{
        pool.load_folder("generation_01/", structure);
    }
    catch(int e) {
        std::cout << "error loading folder number " << e << "\n";
    }

    std::cout.precision(3);
    int num_of_trains = 2;
    int num_of_points = 100;

    for(int i = 0; i < num_of_trains; i++) {
        test_gens = pool.get_testgroup();
        train_gens = pool.get_traingroup();
        double gen_mean_loss = 0;

        for(int j = 0; j < train_gens.size(); j++) {
        // for(int j = 0; j < 1; j++) {
            network.set_parameters(train_gens[j].gen);
            // network.set_parameters(train_gens[0].gen);
            std::cout << "Gen #" << j << "\n\n";
            double mean_loss = 0;
            
            std::cout << "\n";

            // for(double x = 0; x < 1; x += 0.1) {
            for(int k = 0; k < num_of_points; k++) {
                double x = (double) rand()/RAND_MAX;
                inputs[0] = x;
                outputs = network.calculate_outputs(inputs);
                double y = outputs[0], l = loss(x, y);
                std::cout << "x = " << x << "\t|\tf = " << f(x) << "\t|\ty = " << y << "\t|\tl = " << l << "\n";
                mean_loss += l;
            }
            mean_loss /= (double) num_of_points;
            gen_mean_loss += mean_loss;
            std::cout << "\nMean Loss: " << mean_loss << "\n\n";
            
            pool.set_trainfitness(j, mean_loss);
        }

        pool.next_gen();
        
        if(i == num_of_trains - 1) {
            pool.save_folder("generation_02/", structure);
        }
    }

    return EXIT_SUCCESS;
}
