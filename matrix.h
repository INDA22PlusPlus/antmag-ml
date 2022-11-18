
#pragma once
#include<initializer_list>
#include<iostream>
#include<math.h>

template<typename T, int n, int m>
class matrix {
	T Q[n][m];

public:
	matrix() {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				Q[i][j] = 0;
			}
		}
	}

	matrix(std::initializer_list<std::initializer_list<T>> l){
		int i = 0;
		for (auto it = l.begin(); it != l.end(); it++) {
			int j = 0;
			for (auto it2 = (*it).begin(); it2 != (*it).end(); it2++) {
				this->Q[i][j] = (*it2);
				j++;
			}
			i++;
		}
	}

	matrix<T, n, m> operator+(const matrix<T, n, m>& other) const{
		matrix<T, n, m> tmp;

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				tmp.set(i, j, this->get(i, j) + other.get(i, j));
			}
		}
		return tmp;
	}

	matrix<T, n, m> operator-(const matrix<T, n, m>& other) const{
		return (*this) + other*(static_cast<T>(-1));
	}

	matrix<T, n, m> operator*(T a) const{
		matrix<T, n, m> tmp;

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				tmp.set(i, j, this->Q[i][j] * a);
			}
		}
		return tmp;
	}

	template<int ot_m>
	matrix<T, n, ot_m> operator*(const matrix<T, m, ot_m>& other) const{
		matrix<T, n, ot_m> tmp;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < ot_m; j++) {
				T prod = 0;

				for (int k = 0; k < m; k++) {
					prod += this->Q[i][k] * other.get(k, j);
				}
				tmp.set(i, j, prod);
			}
		}
		return tmp;
	}

	//assumes the matrix is either a row vector or a column vector
	double magnitude_squared() const{
		assert(n == 1 || m == 1);
		double r = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				r += this->get(i,j) * this->get(i,j);
			}
		}
		return r;
	}

	matrix<T, m, n> transpose(){
		matrix<T, m, n> res;
		for(int i = 0; i < n; i++){
			for(int j = 0; j < m; j++){
				res.set(j, i, this->get(i,j));
			}
		}
		return res;
	}

	matrix<T, n, m> hadamard_product(const matrix<T, n, m>& other){
		assert(n == 1 || m == 1);
		matrix<T, n, m> res;
		for(int i = 0; i < n; i++){
			for(int j = 0; j < m; j++){
				res.set(i,j, this->get(i,j)*other.get(i,j));
			}
		}
		return res;
	}

	void print_out() {
		std::cout << n << " by " << m << " matrix" << std::endl;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				std::cout << this->Q[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}

	inline T get(int i, int j) const{
		return this->Q[i][j];
	}
	inline void set(int i, int j, T val) {
		this->Q[i][j] = val;
	}
};


