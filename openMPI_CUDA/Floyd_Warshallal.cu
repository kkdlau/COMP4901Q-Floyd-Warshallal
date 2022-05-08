#include <mpi.h>
#include <cmath>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#define is_master (RANK == 0)
#define for_each_block for(int r = 0; r < NUM_BLK_PER_ROWS; r++) for(int c = 0; c < NUM_BLK_PER_ROWS; c++)
int NUM_PROCESS, RANK;
int SIZE_N;
int row_rank, row_size;
int col_rank, col_size;
MPI_Comm row_comm;
MPI_Comm col_comm;



using namespace std::chrono;
using namespace std;
using SMatrix = int**;
using InlineMatrix = int*;

const int DIST_LIMIT = 10000;

bool fread(const string &fpath, int* SIZE_N, SMatrix& M, InlineMatrix& inlined);

void arr_print(InlineMatrix arr, int row_size);

void Fusion_floydWarshall(SMatrix matrix);

void copy_block(SMatrix matrix, int r, int c, int block_width, InlineMatrix dist);


#define CUDA_SAFE_CALL(call)                                              \
	do {                                                                  \
		cudaError_t err = call;                                           \
		if (err != cudaSuccess) {                                         \
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
					__FILE__, __LINE__, cudaGetErrorString(err));         \
			exit(1);                                                      \
		}                                                                 \
	} while (0)

const int TILE_WIDTH = 16;

__global__ void blockwise_min_kernel(int* dist, int* A_ik, int* A_kj, const int row_rank, const int col_rank, const int block_width, const int k);

void write_matrix(SMatrix matrix, int r, int c, int block_width, int* block);

/**
 * defines a 2D Square Matrix with 2D array style.
*/
void malloc_matrix(int width, int**& ptr2d, int*& ptr1d);

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &NUM_PROCESS);
  MPI_Comm_rank(MPI_COMM_WORLD, &RANK);


  SMatrix input_matrix = nullptr;
  InlineMatrix input_inlined = nullptr;


  if (is_master) {
    if (argc < 1)
      return 1;
    if (!fread(string(argv[1]), &SIZE_N, input_matrix, input_inlined)) {
      printf("Error! Please check the format of input file\n");
      return 1;
    }
  }


  SMatrix output_matrix = nullptr;
  InlineMatrix output_inlined = nullptr;

  double p_start;
  if (RANK == 0) {
    p_start = MPI_Wtime();
  }
  // start of the parallel algorithm
  MPI_Bcast(&SIZE_N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // splitting MPI_COMM_WORLD into row blocks and column blocks 
  int row_split = RANK % (int)sqrt(NUM_PROCESS);
  int col_split = RANK / (int)sqrt(NUM_PROCESS);
  MPI_Comm_split(MPI_COMM_WORLD, row_split, RANK, &row_comm);
  MPI_Comm_rank(row_comm, &row_rank);
  MPI_Comm_size(row_comm, &row_size);

  MPI_Comm_split(MPI_COMM_WORLD, col_split, RANK, &col_comm);
  MPI_Comm_rank(col_comm, &col_rank);
  MPI_Comm_size(col_comm, &col_size);

  Fusion_floydWarshall(input_matrix);

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  // end of the parallel algorithm
  MPI_Barrier(MPI_COMM_WORLD);
  if (RANK == 0) {
    double p_end = MPI_Wtime();
    arr_print(input_inlined, SIZE_N);
    cout << "Floyd Warshall openMPI + CUDA Runtime: " << ((p_end - p_start) * 1000) << "ms" << endl;
  }

  // print_matrix(output);
  MPI_Finalize();
  return 0;
}

bool fread(const string &fpath, int *width, SMatrix& M, InlineMatrix& inlined) {
  ifstream fin(fpath.c_str());
  if (!fin.is_open()) {
    return false;
  }
  fin >> (*width);

  malloc_matrix(*width, M, inlined);
  for (int i = 0; i < (*width); i++) {
    for (int j = 0; j < (*width); j++) {
      int v;
      fin >> v;
      if (v > DIST_LIMIT) {
        v = INT_MAX;
      }
      M[i][j] = v;
    }
  }
  fin.close();
  return true;
}

void Fusion_floydWarshall(SMatrix matrix) {
  const int NUM_BLK_PER_ROWS = (int)sqrt(NUM_PROCESS);
  const int BWIDTH = SIZE_N / NUM_BLK_PER_ROWS;

  const int BSIZE = BWIDTH * BWIDTH;
  int* self_buf = new int[BWIDTH * BWIDTH]; // self-block buffer
  vector<int*> send_bufs;
  MPI_Request req;
  if (is_master) {
    for_each_block {
      // printf("master at %d, %d, sending to %d\n", r, c, r * NUM_BLK_PER_ROWS + c);
      int* send_buf = new int[BWIDTH * BWIDTH];
      send_bufs.push_back(send_buf);
      copy_block(matrix, r, c, BWIDTH, send_buf);
      MPI_Isend(send_buf, BSIZE, MPI_INT, r * NUM_BLK_PER_ROWS + c, 0, MPI_COMM_WORLD, &req);
    }
  }
  MPI_Recv(self_buf, BSIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);

  int* rb_buf = new int[BSIZE]{0}; // row-block buffer
  int* cb_buf = new int[BSIZE]{0}; // column-block buffer

  int *dcb_buf, *drb_buf, *d_self;
  cudaMalloc(&dcb_buf, BSIZE * sizeof(int));
  cudaMalloc(&drb_buf, BSIZE * sizeof(int));
  cudaMalloc(&d_self, BSIZE * sizeof(int));
  CUDA_SAFE_CALL(cudaMemcpy(d_self, self_buf, BSIZE * sizeof(int), cudaMemcpyHostToDevice));

  for (int k = 0; k < SIZE_N; k++) {
    // column-group send
    if ((int)(k / BWIDTH) == row_rank) {
      memcpy(rb_buf, self_buf, BSIZE * sizeof(int));
    }
    MPI_Bcast(rb_buf, BSIZE, MPI_INT, (int)(k / BWIDTH), row_comm);

    // row-group send
    if ((int)(k / BWIDTH) == col_rank) {
      memcpy(cb_buf, self_buf, BSIZE * sizeof(int));
    }
    MPI_Bcast(cb_buf, BSIZE, MPI_INT, (int)(k / BWIDTH), col_comm);

    CUDA_SAFE_CALL(cudaMemcpy(drb_buf, rb_buf, BSIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dcb_buf, cb_buf, BSIZE * sizeof(int), cudaMemcpyHostToDevice));

  	dim3 grid((int)ceil(BWIDTH * 1.0 / TILE_WIDTH),
			        (int)ceil(BWIDTH * 1.0 / TILE_WIDTH));
	  dim3 block(TILE_WIDTH, TILE_WIDTH);

    blockwise_min_kernel<<<grid, block>>>(d_self, dcb_buf, drb_buf, row_rank, col_rank, BWIDTH, k);

    CUDA_SAFE_CALL(cudaMemcpy(self_buf, d_self, BSIZE * sizeof(int), cudaMemcpyDeviceToHost));

    MPI_Isend(self_buf, BSIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
    if (is_master) {
      int* recv_buf = new int[BWIDTH * BWIDTH];
      for_each_block {
        MPI_Recv(recv_buf, BSIZE, MPI_INT, r * NUM_BLK_PER_ROWS + c, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        write_matrix(matrix, r, c, BWIDTH, recv_buf);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  delete[] rb_buf;
  delete[] cb_buf;
  delete[] self_buf;
  if (is_master) {
    for(auto ptr: send_bufs) {
      delete[] ptr;
    }
  }
	cudaFree(d_self);
	cudaFree(dcb_buf);
	cudaFree(drb_buf);
}


void arr_print(InlineMatrix arr, int row_size) {
  if (arr == nullptr) {
    printf("nullptr matrix is passed\n");
    return;
  }
  for(int y = 0; y < row_size; y++) {
    for(int x = 0; x < row_size; x++) {
      int e = arr[y * row_size + x];
      if (e > DIST_LIMIT) {
        printf("  %s ", "∞");
      } else {
        printf("%3d ", e);
      }
    }
    printf("\n");
  }
}

void copy_block(SMatrix matrix, int r, int c, int block_width, int* dist) {
  int i = 0;
  r *= block_width;
  c *= block_width;
  for (int _r = 0; _r < block_width; _r++) {
    for (int _c = 0; _c < block_width; _c++) {
      dist[i++] = matrix[_r + r][_c + c];
    }
  }
}

__global__ void blockwise_min_kernel(int* dist, int* A_ik, int* A_kj, const int row_rank, const int col_rank, const int block_width, const int k) {
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int i = by * TILE_WIDTH + ty;
	int j = bx * TILE_WIDTH + tx;

  int prev;
  int aik;
  int akj;

  if (i < block_width && j < block_width) {
    prev = dist[i * block_width + j];
    aik = A_ik[i * block_width + (k % block_width)];
    akj = A_kj[(k % block_width) * block_width + j];

    if (prev > (aik + akj) && (akj <= DIST_LIMIT && aik <= DIST_LIMIT)) {
      dist[i * block_width + j] = aik + akj;
    }
  }
}


void write_matrix(SMatrix matrix, int r, int c, int block_width, int* block) {
  if (matrix == nullptr) {
    printf("nullptr matrix is passed\n");
    return;
  }
  int i = 0;
  for (int _r = 0; _r < block_width; _r++) {
    for (int _c = 0; _c < block_width; _c++) {
      matrix[_r + r * block_width][_c + c * block_width] = block[i++];
    }
  }
}

void malloc_matrix(int width, int**& ptr2d, int*& ptr1d) {
  ptr1d = new int[width * width];
  ptr2d = new int*[width];
  for(int i = 0; i < width; i++) {
    ptr2d[i] = ptr1d + i * width;
  }
}