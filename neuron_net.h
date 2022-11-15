#pragma once

#include<vector>
#include <cstdarg>
#include<assert.h>
#include<random>
#include"matrix.h"


struct node {
	int label;
};

enum LAYER
{
	INPUT_LAYER = 0,
	HIDDEN_LAYER = 1,
	OUTPUT_LAYER = 2
};

class neuron_net {
	static const size_t INPUT_L_CNT = 4;
	static const size_t HIDDEN_L1_CNT = 2;
	static const size_t OUTPUT_L_CNT = 2;

	matrix<double, neuron_net::INPUT_L_CNT, neuron_net::HIDDEN_L1_CNT> layer1;
	matrix<double, neuron_net::HIDDEN_L1_CNT, neuron_net::OUTPUT_L_CNT> layer2;

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

	void setup_dense() {
		for (int i = 0; i < neuron_net::INPUT_L_CNT; i++) {
			for (int j = 0; j < neuron_net::HIDDEN_L1_CNT; j++) {
				this->connect_random_weight(LAYER::INPUT_LAYER, LAYER::HIDDEN_LAYER, i, j);
			}
		}

		for (int i = 0; i < neuron_net::HIDDEN_L1_CNT; i++) {
			for (int j = 0; j < neuron_net::OUTPUT_L_CNT; j++) {
				this->connect_random_weight(LAYER::HIDDEN_LAYER, LAYER::OUTPUT_LAYER, i, j);
			}
		}
	}

	int check(matrix<double, 1, neuron_net::INPUT_L_CNT> input_vec) {
		matrix<double, 1, neuron_net::HIDDEN_L1_CNT> h_layer_1_vals = input_vec * this->layer1;
		matrix<double, 1, neuron_net::OUTPUT_L_CNT> output_layer_vals = h_layer_1_vals * this->layer2;

		std::pair<double, int> best;
		for (int i = 0; i < neuron_net::OUTPUT_L_CNT; i++) {
			double c = output_layer_vals.get(0, i);
			if (best.first < c) {
				best.first = c;
				best.second = i;
			}
		}
		return best.second;
	}



};

