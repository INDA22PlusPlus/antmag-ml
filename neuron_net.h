#pragma once

#include<vector>
#include<tuple>
#include <cstdarg>
#include<math.h>
#include<assert.h>
#include<random>
#include"matrix.h"

/*

13 13 13 13

52! / 

13 13 13 13

(p, q, r, s)

p, q, r, s

p, 



*/

struct node {
	int label;
};

enum LAYER
{
	INPUT_LAYER = 0,
	HIDDEN_LAYER = 1,
	OUTPUT_LAYER = 2
};

const size_t INPUT_L_CNT = 28*28;
const size_t HIDDEN_L1_CNT = 28*3;
const size_t OUTPUT_L_CNT = 10;
const size_t L_CNT = 3;
constexpr size_t MAX_L_CNT = std::max(INPUT_L_CNT, std::max(HIDDEN_L1_CNT, OUTPUT_L_CNT));
const size_t BATCH_SIZE = 10;

struct net_params{
	matrix<double, 1, HIDDEN_L1_CNT> b1;
	matrix<double, 1, OUTPUT_L_CNT> b2;
	matrix<double, INPUT_L_CNT, HIDDEN_L1_CNT> w1;
	matrix<double, HIDDEN_L1_CNT, OUTPUT_L_CNT> w2;
};

struct training_input{
	matrix<double, 1, INPUT_L_CNT> input_vector;
	matrix<double, 1, OUTPUT_L_CNT> expected_vector;
};

struct layer_vectors{
	matrix<double, 1, OUTPUT_L_CNT> output_layer;
	matrix<double, 1, HIDDEN_L1_CNT> hidden_layer;
};

class neuron_net {
	matrix<double, INPUT_L_CNT, HIDDEN_L1_CNT> layer1;
	matrix<double, HIDDEN_L1_CNT, OUTPUT_L_CNT> layer2;
	matrix<double, 1, HIDDEN_L1_CNT> bias1;
	matrix<double, 1, OUTPUT_L_CNT> bias2;
	
public:

	neuron_net() {}
	void connect(LAYER from, LAYER to, size_t from_label, size_t to_label, double w) {
		assert(from < to);
		if (from == LAYER::INPUT_LAYER) {
			this->layer1.set(from_label, to_label, w);
		}
		else {
			this->layer2.set(from_label, to_label, w);
		}
	}

	void connect_random_weight(LAYER from, LAYER to, size_t from_label, size_t to_label) {
		double w = (double(rand()) / double(RAND_MAX)) * 0.6 - 0.3;
		this->connect(from, to, from_label, to_label, w);
	}
	
	void create_random_biases(){
		for(int j = 0; j < HIDDEN_L1_CNT; j++){
			double b =  (double(rand()) / double(RAND_MAX)) * 0.6 - 0.3;
			bias1.set(0, j, b);
		}
		for(int j = 0; j < OUTPUT_L_CNT; j++){
			double b =  (double(rand()) / double(RAND_MAX)) * 0.6 - 0.3;
			bias2.set(0, j, b);
		}
	}

	void setup_dense() {
		for (int i = 0; i < INPUT_L_CNT; i++) {
			for (int j = 0; j < HIDDEN_L1_CNT; j++) {
				this->connect_random_weight(LAYER::INPUT_LAYER, LAYER::HIDDEN_LAYER, i, j);
			}
		}

		for (int i = 0; i < HIDDEN_L1_CNT; i++) {
			for (int j = 0; j < OUTPUT_L_CNT; j++) {
				this->connect_random_weight(LAYER::HIDDEN_LAYER, LAYER::OUTPUT_LAYER, i, j);
			}
		}
	}

	void shuffle(){
		this->setup_dense();
		this->create_random_biases();
	}

	layer_vectors calculate_activations(matrix<double, 1, INPUT_L_CNT> input_vec) const{
		matrix<double, 1, HIDDEN_L1_CNT> layer2_activation = this->vectorize_activation(input_vec * this->layer1 + this->bias1);
		matrix<double, 1, OUTPUT_L_CNT> layer3_activation = this->vectorize_activation(layer2_activation * this->layer2 + this->bias2);

		return {layer3_activation, layer2_activation};
	}

	std::pair<matrix<double, 1, OUTPUT_L_CNT>, matrix<double, 1, HIDDEN_L1_CNT>> get_weighted_inputs(const matrix<double, 1, INPUT_L_CNT>& input_vec) const{
		matrix<double, 1, HIDDEN_L1_CNT> intermediate = input_vec * this->layer1 + this->bias1; //checked
		return {this->vectorize_activation(intermediate) * this->layer2 + this->bias2, intermediate}; //Should also be correct
	}

	int check(matrix<double, 1, INPUT_L_CNT> input_vec) const{
		auto activations = this->calculate_activations(input_vec);
		
		std::pair<double, int> best;
		for (int i = 0; i < OUTPUT_L_CNT; i++) {
			double c = activations.output_layer.get(0, i);
			if (best.first < c) {
				best.first = c;
				best.second = i;
			}
		}
		return best.second;
	}

	//We do the quadratic cost function
	double cost(matrix<double, 1, INPUT_L_CNT>& input_vec, matrix<double, 1, OUTPUT_L_CNT>& target){
		auto activations = this->calculate_activations(input_vec);
		return (target - activations.output_layer).magnitude_squared() / 2.0;
	}

	inline double activation_func(double x) const{
		return 1.0 / (1.0 + exp(-x)); 
	}
	inline double activation_func_derivative(double x) const{
		return activation_func(x) * (1 - activation_func(x));
	}	

	template<int m>
	matrix<double, 1, m> vectorize_activation_derivative(matrix<double, 1, m> A) const{
		for(int j = 0; j < m; j++){
			A.set(0, j, this->activation_func_derivative(A.get(0,j)));
		}
		return A;
	}

	template<int m>
	matrix<double, 1, m> vectorize_activation(matrix<double, 1, m> A) const{
		for(int j = 0; j < m; j++){
			A.set(0,j, this->activation_func(A.get(0,j)));
		}
		return A;
	}

	layer_vectors back_propagation(const training_input& ti){
		auto activations = this->calculate_activations(ti.input_vector);
		auto input_weights = this->get_weighted_inputs(ti.input_vector);

		matrix<double, 1, OUTPUT_L_CNT> neuron_error_3 = 
		(activations.output_layer - ti.expected_vector).hadamard_product(this->vectorize_activation_derivative(input_weights.first));
		
		/*The construction of the weigth matrix in the formula is equal to mine but transposed*/
		matrix<double, 1, HIDDEN_L1_CNT> neuron_error_2 = 
		(this->layer2 * neuron_error_3.transpose()).transpose().hadamard_product(this->vectorize_activation_derivative(input_weights.second));

		return {neuron_error_3, neuron_error_2};
	}

	net_params calculate_gradient(training_input ti){
		layer_vectors error = this->back_propagation(ti);
		layer_vectors activations = this->calculate_activations(ti.input_vector);
		net_params gradient;

		//Configuring the bias gradients
		gradient.b1 = error.hidden_layer;
		gradient.b2 = error.output_layer;
		
		//Configuring the weight gradients
		for(int i = 0; i < INPUT_L_CNT; i++){
			for(int j = 0; j < HIDDEN_L1_CNT; j++){
				gradient.w1.set(i,j, error.hidden_layer.get(0,j) * ti.input_vector.get(0,i));
			}
		}
		for(int i = 0; i < HIDDEN_L1_CNT; i++){
			for(int j = 0; j < OUTPUT_L_CNT; j++){
				gradient.w2.set(i,j, error.output_layer.get(0,j) * activations.hidden_layer.get(0,i));
			}
		}
		return gradient;
	}

	void train(const training_input& ti){
		net_params gradient = this->calculate_gradient(ti);
		this->update_net(gradient);
	}

	void update_net(net_params gradient){
		double learning_rate = 0.1;	
		this->bias1 = this->bias1 - gradient.b1 * learning_rate;
		this->bias2 = this->bias2 - gradient.b2 * learning_rate;
		this->layer1 = this->layer1 - gradient.w1 * learning_rate;
		this->layer2 = this->layer2 - gradient.w2 * learning_rate;
	}

	static training_input format_data(unsigned char pixel_buffer[28*28], unsigned char label){
		training_input ti;
		for(int j = 0; j < 28*28; j++){
			ti.input_vector.set(0, j, static_cast<double>(pixel_buffer[j]) / 255.0);
		}
		ti.expected_vector.set(0, static_cast<int>(label), 1.0);
		return ti;
	}

	//pixel_buffer, and label_buffer denotes the start
	void excersize_batch(unsigned char* pixel_buffer, unsigned char* label_buffer){
		net_params gradient_sum;
		for(int i = 0; i < BATCH_SIZE; i++){
			training_input inp = this->format_data(pixel_buffer + i*28*28, *(label_buffer + i));
			net_params this_gradient = this->calculate_gradient(inp);
			gradient_sum.b1 = gradient_sum.b1 + this_gradient.b1;
			gradient_sum.b2 = gradient_sum.b2 + this_gradient.b2;
			gradient_sum.w1 = gradient_sum.w1 + this_gradient.w1;
			gradient_sum.w2 = gradient_sum.w2 + this_gradient.w2;
		}
		this->update_net(gradient_sum);
	}

	//True if correct
	bool test(unsigned char pixel_buffer[28*28], unsigned char label) const{
		training_input ti = format_data(pixel_buffer, label);
		int res = this->check(ti.input_vector);
		return static_cast<int>(label) == res;
	}


	void print_weight_matrices(){
		this->layer1.print_out();
		this->layer2.print_out();
		std::cout << "-------------BIAS------------" << std::endl;
		this->bias1.print_out();
		this->bias2.print_out();
	}
};

