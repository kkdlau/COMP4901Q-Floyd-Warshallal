#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std::chrono;

using namespace std;
using SMatrix = vector<vector<int>>; // Square Matrix

const int DIST_LIMIT = 10000;

bool fread(const string &fpath, int *SIZE_N, SMatrix *M);

void inline_vector(int *&arr, const SMatrix &m);

void SMatrix_print(const SMatrix &m);

#define print_matrix(matrix)                                                   \
  printf("%s =\n", #matrix);                                                   \
  SMatrix_print(matrix);

void MP_floydWarshall(const SMatrix &m, SMatrix &output);

int main(int argc, char **argv) {
  if (argc < 1)
    return 1;

  SMatrix input;
  SMatrix output;
  int size;
  if (!fread(string(argv[1]), &size, &input)) {
    printf("Error! Please check the format of input file\n");
    return 1;
  }

  // print_matrix(input);

  auto start = high_resolution_clock::now();

  MP_floydWarshall(input, output);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);

  cout << "Floyd Warshall openMP Runtime: " << (duration.count() / 1000.0)
       << "ms" << endl;

  // print_matrix(output);

  return 0;
}

bool fread(const string &fpath, int *SIZE_N, SMatrix *M) {
  ifstream fin(fpath.c_str());
  if (!fin.is_open()) {
    return false;
  }
  fin >> (*SIZE_N);
  *M = SMatrix(*SIZE_N, vector<int>(*SIZE_N, 0));
  for (int i = 0; i < (*SIZE_N); i++) {
    for (int j = 0; j < (*SIZE_N); j++) {
      int v;
      fin >> v;
      if (v > DIST_LIMIT) {
        v = INT_MAX;
      }
      (*M)[i][j] = v;
    }
  }
  fin.close();
  return true;
}

void inline_vector(int *&arr, const SMatrix &m) {
  if (arr != nullptr)
    delete[] arr;
  arr = new int[m.size() * m.size()];
  unsigned int i = 0;
  for (const auto &row : m) {
    for (const auto &e : row) {
      arr[i++] = e;
    }
  }
}

void MP_floydWarshall(const SMatrix &m, SMatrix &output) {
  int i, j, k;
  const int SIZE = m.size();
  output = m;

  for (k = 0; k < SIZE; k++) {
    #pragma omp parallel for collapse(2)
    for (i = 0; i < SIZE; i++) {
      for (j = 0; j < SIZE; j++) {
        if (output[i][j] > (output[i][k] + output[k][j]) &&
            (output[k][j] <= DIST_LIMIT && output[i][k] <= DIST_LIMIT))
          output[i][j] = output[i][k] + output[k][j];
      }
    }
  }
}

void SMatrix_print(const SMatrix &m) {
  for (const auto &row : m) {
    for (const auto &e : row) {
      if (e > DIST_LIMIT) {
        printf("  %s ", "∞");
      } else {
        printf("%3d ", e);
      }
    }
    printf("\n");
  }
}