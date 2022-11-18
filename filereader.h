
#pragma once
#include<fstream>
#include<iostream>

namespace file_reader{
    uint32_t read_int_from_binary(std::ifstream& fstrm){
        uint32_t v;
        unsigned char b;
        fstrm >> b; v = b;
        fstrm >> b; v = (v << 8) | b;
        fstrm >> b; v = (v << 8) | b;
        fstrm >> b; v = (v << 8) | b;
        return v;
    }

    char read_char_from_binary(std::ifstream& fstrm){
        unsigned char b;
        fstrm >> b;
        return b;
    }

    int read_pixel_data(int n, unsigned char* p, bool for_testing){
        std::ifstream stream;

        if(for_testing){
            stream.open("MNIST_txt/t10k-images.idx3-ubyte", std::ios::binary);
        }else{
            stream.open("MNIST_txt/train-images.idx3-ubyte", std::ios::binary);
        }        
        if(!stream){
            std::cout << "Failed to open file!" << std::endl;
            return 1;
        }

        uint32_t magic_num = read_int_from_binary(stream);
        uint32_t img_cnt = read_int_from_binary(stream);
        uint32_t row_cnt = read_int_from_binary(stream);
        uint32_t col_cnt = read_int_from_binary(stream);
        
        int r = std::min(60'000, n);
        for(int i = 0; i < r; i++){
            p[i] = read_char_from_binary(stream);
        }
        stream.close();
        return 0;
    }

    int read_label_data(int n, unsigned char* p, bool for_testing){
        std::ifstream stream; 
        if(for_testing){
            stream.open("MNIST_txt/t10k-labels.idx1-ubyte", std::ios::binary);
        }else{
            stream.open("MNIST_txt/train-labels.idx1-ubyte", std::ios::binary);
        }
        if(!stream){
            std::cout << "Failed to open file!" << std::endl;
            return 1;
        }

        uint32_t magic_num = read_int_from_binary(stream);
        uint32_t label_cnt = read_int_from_binary(stream);

        int r = std::min(60'000, n);
        for(int i = 0; i < r; i++){
            p[i] = read_char_from_binary(stream);
        }
        stream.close();
        return 0;
    }




}
