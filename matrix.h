
#pragma once
#include<initializer_list>
#include<math.h>

template<typename T, int n, int m>
class matrix {
	T Q[n][m];

public:
	matrix() {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				Q[i][j] = (i == j);
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

	matrix<T, n, m> operator+(const matrix<T, n, m>& other) {
		matrix<T, n, m> tmp;

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				tmp.set(i, j, this->get(i, j) + other.get(i, j));
			}
		}
		return tmp;
	}

	matrix<T, n, m> operator*(int a) {
		matrix<T, n, m> tmp;

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				tmp.set(i, j, this->Q[i][j] * a);
			}
		}
		return tmp;
	}

	template<int ot_m>
	matrix<T, n, ot_m> operator*(const matrix<T, m, ot_m>& other) {
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


