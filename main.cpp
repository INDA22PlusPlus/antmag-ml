
#include"neuron_net.h"
#include"matrix.h"
#include"filereader.h"
#include<iostream>

using namespace std;

const int EXCERSIZE_CNT = 35000;
const int TEST_CNT = 3000;
unsigned char pixel_buffer[28*28*EXCERSIZE_CNT];
unsigned char label_buffer[EXCERSIZE_CNT];

void read_and_excersize(neuron_net& nn){
    file_reader::read_pixel_data(28*28*EXCERSIZE_CNT, pixel_buffer, false);
    file_reader::read_label_data(EXCERSIZE_CNT, label_buffer, false);
    
    for(int i = 0; i < EXCERSIZE_CNT/BATCH_SIZE; i++){
        nn.excersize_batch(pixel_buffer + 28*28*i*BATCH_SIZE, label_buffer + i*BATCH_SIZE);
    }
}

void read_and_test(const neuron_net& nn){
    file_reader::read_pixel_data(28*28*TEST_CNT, pixel_buffer, true);
    file_reader::read_label_data(TEST_CNT, label_buffer, true);
    int r = 0;
    for(int i = 0; i < TEST_CNT; i++){
        r += nn.test(pixel_buffer + i*28*28, label_buffer[i]);
    }

    cout << "Accuracy: " << double(r) / double(TEST_CNT) * 100.0 << "%" << endl;
}

/*
void simple_system_example(){
    for(int i = 0; i < 8000000; i++){
        nn.train({{double(rand()) / double(RAND_MAX)}}, {{0}});
    }
    nn.print_weight_matrices();

    double error_sum = 0;
    for(int i = 0; i < 10000; i++){
        matrix<double,1,1> inp = {{double(rand()) / double(RAND_MAX)}};
        matrix<double,1,1> t = {{0}};
        error_sum += nn.cost(inp, t);
    }
    cout << error_sum / double(10000) << endl;
}
*/

int main(){
    srand(time(NULL));

    neuron_net nn;
    nn.shuffle();

    matrix<double, 1, 1> input_vec = {
        {3.0},
    };
    
    read_and_excersize(nn);
    read_and_test(nn);
}

