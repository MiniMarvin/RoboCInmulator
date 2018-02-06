/**
 * Problem: The routines of starting the match and so on are problematic because they need a time 
 * to stop the simulation an just them being able to restart, because of that, make the system recover
 * the handlers everytime it's going to restart the simulation because it's enought time to the system
 * restar the simulation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <thread>
#include "../libs/simulation_group.hpp"
#include "../libs/robot_agent.hpp"

// System includes
#include <time.h>
#include <cstdlib>
#include <string>
#include <exception>

// OpenNN includes
#include <opennn.h>
#include "../libs/genetic_pool.hpp"

extern "C" {
#include "extApi.h"
	/*  #include "extApiCustom.h" if you wanna use custom remote API functions! */
}

#define MAX_SPEED 30

using namespace std;
using namespace OpenNN;


// global variables
clock_t this_time;
clock_t last_time;
const int NUM_SECONDS = 5;
double time_counter;


double normalize(double value, double limit) {
	value *= limit;
	if(value < 0) value = 0;
	if(value > limit) value = limit;
	return value;
}

bool is_timeout() {
	this_time = clock();
	time_counter += (double)(this_time - last_time);
	return (time_counter > (double)(NUM_SECONDS * CLOCKS_PER_SEC));
}

void reset_timer() {
	this_time = last_time = clock();
	time_counter = 0;
}

void print_positions(simulationGroup sim) {
	vector<Position> teamA = sim.get_teamAPos();
	vector<Position> teamB = sim.get_teamBPos();
	Position ball = sim.get_ballPos();

	cout << "Ball" << "\t";
	cout << "ball " << " at -> " << "x: " << ball.x << " y: " << 
				ball.y << " z: " << ball.z << " t: " << ball.t << endl;

	cout << "Team A" << "\t";
	for (int i = 0; i < teamA.size(); ++i) {
		cout << "robot " << i << " at -> " << "x: " << teamA[i].x << " y: " << 
				teamA[i].y << " z: " << teamA[i].z << " t: " << teamA[i].t << endl;
	}

	cout << "Team B " << "\t";
	for (int i = 0; i < teamB.size(); ++i) {
		cout << "robot " << i << " at -> " << "x: " << teamB[i].x << " y: " << 
				teamB[i].y << " z: " << teamB[i].z << " t: " << teamB[i].t << endl;
	}
	cout << endl << endl;
}

void setup_pos(simulationGroup sim) {
	Position start_ball = {0, 0, 0, 0};
	Position start_robot1 = {0.1, 0, 0, 0};
	Position start_robot2 = {-0.1, 0, 0, 0};
	vector<Position> teamAPos;
	vector<Position> teamBPos;

	teamAPos.push_back(start_robot1);
	teamBPos.push_back(start_robot2);
	
	sim.setup_startPos(teamAPos, teamBPos, start_ball);
}

int main() {
	string robots[] = {"RobotFrame", "RobotFrame0"};
	string motors[]= {"LeftMotor_1", "RightMotor_1", "LeftMotor_2", "RightMotor_2"};
	vector<string> teamA(robots, robots + 1);
	vector<string> teamB(robots + 1, robots + 2);
	string ball = "Bola";
	simulationGroup sim(teamA, teamB, ball);

	srand(time(NULL));

	// Motor handlers -------------------------------------------//
	RobotAgent robot1(robots[0], motors[0], motors[1], sim);
	RobotAgent robot2(robots[1], motors[2], motors[3], sim);

	robot1.set_team(1);
	robot2.set_team(2);
	//-----------------------------------------------------------//
	
	// Setup the Genetic Algoirithm agents ----------------------//
	// int nums[] = {8, 30, 30, 30, 30, 30, 2};
	// int nums[] = {8, 30, 30, 30, 30, 30, 2};
	int nums[] = {8, 50, 50, 50, 50, 50, 50, 50, 50, 2};
    Vector<int> structure(10, 0);
    
    for(int i = 0; i < 10; i++) {
        structure[i] = nums[i];
    }

    Vector<double> inputs(8, 0.0);
    Vector<double> outputs(2, 0.0);
    Gene buffgen;

    MultilayerPerceptron tester(structure);
    MultilayerPerceptron trainer(structure);

    geneticPool pool;

    std::vector<Gene> test_gens;
    test_gens = pool.random_gens(100, structure);
    std::vector<Gene> train_gens;
    train_gens = test_gens;
    // pool.load_folder("player_gens", structure);
    // pool.load_folder("game_gens", structure);
    pool.load_folder("10_layers", structure);

    // pool.set_traingen(train_gens);
    // pool.set_testgen(test_gens);
    test_gens = pool.get_testgroup();
    train_gens = pool.get_traingroup();

    // for(int i = 0; i < test_gens.size(); i++) {
    // 	pool.set_testfitness(i, 0);
    // }
    


    int num_of_trains = 200;

	cout << "Start of the system simulation!" << endl;
	for(int i = 0; i < num_of_trains && sim.clientID != -1; i++) {
		double mutate_rate = ((double)rand())/(3.0*RAND_MAX);
		pool.set_mutationrate(mutate_rate);
		mutate_rate = ((double)rand())/(3.0*RAND_MAX);
		pool.set_crossoverrate(mutate_rate);

		cout << "Startin the simulation for the " << i << "nd time!" << endl;

		// Setup the gens group -------------------------------------//
		test_gens = pool.get_testgroup();
        train_gens = pool.get_traingroup();
        double fitness = 0;
        int should_reset = 1;
        double ball_diff = 0;
        cout << "------------------------------------------------------------" << endl;
        cout << "Generation #" << i << endl;

        for(int j = 0; j < train_gens.size();) {
	        buffgen = pool.get_testgen();
	        tester.set_parameters(buffgen.gen);
	        trainer.set_parameters(train_gens[j].gen);

			// Motor handlers -------------------------------------------//
			robot1.set_motors(motors[0], motors[1]);
			robot2.set_motors(motors[2], motors[3]);
			//-----------------------------------------------------------//
			
			// Start position -------------------------------------------//
			setup_pos(sim);
			sim.start();
			//-----------------------------------------------------------//
	        
			// Reset timer for allow the program for run for n seconds
			if (should_reset == 1) {
        		reset_timer();
        		should_reset = 0;
        		fitness = 0;
			}

			// Region for gen test
			// for(int i = 0; i < 30 && simxGetConnectionId(sim.clientID) != -1; i++) {
			while(!is_timeout() && simxGetConnectionId(sim.clientID) != -1) {

				// Setup the network activation
				vector<Position> teamA = sim.get_teamAPos();
				vector<Position> teamB = sim.get_teamBPos();
				Position a_pos = teamA[0];
				Position b_pos = teamB[0];
				Position ball_pos = sim.get_ballPos();
				double a_val1 = 0;
				double a_val2 = 0;
				double b_val1 = 0;
				double b_val2 = 0;

				inputs[0] = ball_pos.x/BOARD_X_SIZE;
				inputs[1] = ball_pos.y/BOARD_Y_SIZE;
				// Mine position
				inputs[2] = a_pos.x/BOARD_X_SIZE;
				inputs[3] = a_pos.y/BOARD_Y_SIZE;
				inputs[4] = a_pos.t;
				// Opponent position
				inputs[5] = b_pos.x/BOARD_X_SIZE;
				inputs[6] = b_pos.y/BOARD_Y_SIZE;
				inputs[7] = b_pos.t;

				ball_diff = -(abs(inputs[0] - inputs[2]) + 
								abs(inputs[1] - inputs[3]));
				fitness += ball_diff;

				outputs = trainer.calculate_outputs(inputs);
				a_val1 = normalize(outputs[0], MAX_SPEED);
				a_val2 = normalize(outputs[1], MAX_SPEED);


				inputs[0] = 1 - ball_pos.x/BOARD_X_SIZE;
				inputs[1] = 1 - ball_pos.y/BOARD_Y_SIZE;
				// Mine position
				inputs[5] = 1 - b_pos.x/BOARD_X_SIZE;
				inputs[6] = 1 - b_pos.y/BOARD_Y_SIZE;
				inputs[7] = 1 - b_pos.t;
				// Opponent position
				inputs[2] = 1 - a_pos.x/BOARD_X_SIZE;
				inputs[3] = 1 - a_pos.y/BOARD_Y_SIZE;
				inputs[4] = 1 - a_pos.t;

				outputs = tester.calculate_outputs(inputs);
				b_val1 = normalize(outputs[0], MAX_SPEED);
				b_val2 = normalize(outputs[1], MAX_SPEED);

				// setup velocities
				// robot1.set_power(rand()%20, rand()%20);
				// robot2.set_power(rand()%20, rand()%20);

				robot1.set_power(a_val1, a_val2);
				robot2.set_power(b_val1, b_val2);

				if (sim.is_goal() != 0) {
					cout << "GOAL!!!!" << endl;
					fitness += 100000*sim.is_goal();
					break;
				}

				// print_positions(sim);
				extApi_sleepMs(2);
			}

			if(is_timeout()) {
        		cout << "------------------------------------------------------------" << endl;
				cout << "Fitness of gen #" << j << ": " << fitness << "\n";
				pool.set_trainfitness(j, fitness);
				j++;
				should_reset = 1;
				fitness = 0;
			}

			// setup velocities -----------------------------------------//
			robot1.set_power(0,0);
			robot2.set_power(0,0);
			//-----------------------------------------------------------//
			

			cout << "Stoping the simmulation" << endl;
			setup_pos(sim);
			sim.stop();
			// std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        pool.set_mutationrate((rand()%20)/100.0);
        pool.next_gen();
        // pool.save_folder("player_gens", structure);
        // pool.save_folder("game_gens", structure);
        pool.save_folder("10_layers", structure);
	}

	cout << "Finishing communication" << endl;
	setup_pos(sim);
	sim.endCom();

	return EXIT_SUCCESS;
}
