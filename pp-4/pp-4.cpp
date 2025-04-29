#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <mpi.h>

using namespace std;
using Matrix = vector<int>;

const int STRASSEN_THRESHOLD = 64;

Matrix generateRandomMatrix(int n) {
    Matrix A(n * n);
    for (int i = 0; i < n * n; ++i)
        A[i] = rand() % 10;
    return A;
}

Matrix standardMultiply(const Matrix& A, const Matrix& B, int n) {
    Matrix C(n * n, 0);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
    return C;
}

Matrix add(const Matrix& A, const Matrix& B, int n) {
    Matrix C(n * n);
    for (int i = 0; i < n * n; ++i)
        C[i] = A[i] + B[i];
    return C;
}

Matrix subtract(const Matrix& A, const Matrix& B, int n) {
    Matrix C(n * n);
    for (int i = 0; i < n * n; ++i)
        C[i] = A[i] - B[i];
    return C;
}

Matrix getSubMatrix(const Matrix& M, int originalSize, int x, int y, int subSize) {
    Matrix sub(subSize * subSize);
    for (int i = 0; i < subSize; ++i)
        for (int j = 0; j < subSize; ++j)
            sub[i * subSize + j] = M[(x + i) * originalSize + (y + j)];
    return sub;
}

Matrix strassenSequential(const Matrix& A, const Matrix& B, int n) {
    if (n <= STRASSEN_THRESHOLD)
        return standardMultiply(A, B, n);

    int newSize = n / 2;
    Matrix A11 = getSubMatrix(A, n, 0, 0, newSize);
    Matrix A12 = getSubMatrix(A, n, 0, newSize, newSize);
    Matrix A21 = getSubMatrix(A, n, newSize, 0, newSize);
    Matrix A22 = getSubMatrix(A, n, newSize, newSize, newSize);

    Matrix B11 = getSubMatrix(B, n, 0, 0, newSize);
    Matrix B12 = getSubMatrix(B, n, 0, newSize, newSize);
    Matrix B21 = getSubMatrix(B, n, newSize, 0, newSize);
    Matrix B22 = getSubMatrix(B, n, newSize, newSize, newSize);

    Matrix M1 = strassenSequential(add(A11, A22, newSize), add(B11, B22, newSize), newSize);
    Matrix M2 = strassenSequential(add(A21, A22, newSize), B11, newSize);
    Matrix M3 = strassenSequential(A11, subtract(B12, B22, newSize), newSize);
    Matrix M4 = strassenSequential(A22, subtract(B21, B11, newSize), newSize);
    Matrix M5 = strassenSequential(add(A11, A12, newSize), B22, newSize);
    Matrix M6 = strassenSequential(subtract(A21, A11, newSize), add(B11, B12, newSize), newSize);
    Matrix M7 = strassenSequential(subtract(A12, A22, newSize), add(B21, B22, newSize), newSize);

    Matrix C(n * n);
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            int idx = i * newSize + j;
            C[i * n + j] = M1[idx] + M4[idx] - M5[idx] + M7[idx];
            C[i * n + j + newSize] = M3[idx] + M5[idx];
            C[(i + newSize) * n + j] = M2[idx] + M4[idx];
            C[(i + newSize) * n + j + newSize] = M1[idx] - M2[idx] + M3[idx] + M6[idx];
        }
    }
    return C;
}

Matrix strassenMPI(Matrix& A, Matrix& B, int n, int rank, int size) {
    if (n <= STRASSEN_THRESHOLD)
        return standardMultiply(A, B, n);

    int newSize = n / 2;
    Matrix A11 = getSubMatrix(A, n, 0, 0, newSize);
    Matrix A12 = getSubMatrix(A, n, 0, newSize, newSize);
    Matrix A21 = getSubMatrix(A, n, newSize, 0, newSize);
    Matrix A22 = getSubMatrix(A, n, newSize, newSize, newSize);

    Matrix B11 = getSubMatrix(B, n, 0, 0, newSize);
    Matrix B12 = getSubMatrix(B, n, 0, newSize, newSize);
    Matrix B21 = getSubMatrix(B, n, newSize, 0, newSize);
    Matrix B22 = getSubMatrix(B, n, newSize, newSize, newSize);

    Matrix M_data_A[7], M_data_B[7];
    M_data_A[0] = add(A11, A22, newSize); M_data_B[0] = add(B11, B22, newSize);
    M_data_A[1] = add(A21, A22, newSize); M_data_B[1] = B11;
    M_data_A[2] = A11;                    M_data_B[2] = subtract(B12, B22, newSize);
    M_data_A[3] = A22;                    M_data_B[3] = subtract(B21, B11, newSize);
    M_data_A[4] = add(A11, A12, newSize); M_data_B[4] = B22;
    M_data_A[5] = subtract(A21, A11, newSize); M_data_B[5] = add(B11, B12, newSize);
    M_data_A[6] = subtract(A12, A22, newSize); M_data_B[6] = add(B21, B22, newSize);

    Matrix M_results[7];
    for (int i = 0; i < 7; ++i) {
        int dest = i + 1;
        if (dest < size) {
            MPI_Send(&newSize, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(M_data_A[i].data(), newSize * newSize, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(M_data_B[i].data(), newSize * newSize, MPI_INT, dest, 2, MPI_COMM_WORLD);
        }
        else {
            M_results[i] = strassenSequential(M_data_A[i], M_data_B[i], newSize);
        }
    }

    for (int i = 0; i < 7; ++i) {
        int src = i + 1;
        if (src < size) {
            M_results[i].resize(newSize * newSize);
            MPI_Recv(M_results[i].data(), newSize * newSize, MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    Matrix C(n * n);
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            int idx = i * newSize + j;
            C[i * n + j] = M_results[0][idx] + M_results[3][idx] - M_results[4][idx] + M_results[6][idx];
            C[i * n + j + newSize] = M_results[2][idx] + M_results[4][idx];
            C[(i + newSize) * n + j] = M_results[1][idx] + M_results[3][idx];
            C[(i + newSize) * n + j + newSize] = M_results[0][idx] - M_results[1][idx] + M_results[2][idx] + M_results[5][idx];
        }
    }
    return C;
}

void worker(int rank) {
    while (true) {
        int newSize;
        MPI_Recv(&newSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (newSize <= 0) break;

        Matrix A_part(newSize * newSize), B_part(newSize * newSize);
        MPI_Recv(A_part.data(), newSize * newSize, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_part.data(), newSize * newSize, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Matrix result = strassenSequential(A_part, B_part, newSize);
        MPI_Send(result.data(), newSize * newSize, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }
}

bool validateMatrices(const Matrix& A, const Matrix& B, int n) {
    for (int i = 0; i < n * n; ++i)
        if (A[i] != B[i]) return false;
    return true;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1024;
    if (argc >= 2) n = atoi(argv[1]);

    Matrix A, B, C_std, C_seq, C_par;
    if (rank == 0) {
        srand(time(0));
        A = generateRandomMatrix(n);
        B = generateRandomMatrix(n);

        auto start = chrono::high_resolution_clock::now();
        C_std = standardMultiply(A, B, n);
        auto end = chrono::high_resolution_clock::now();
        cout << "Standard: " << chrono::duration<double, milli>(end - start).count() << " ms\n";

        start = chrono::high_resolution_clock::now();
        C_seq = strassenSequential(A, B, n);
        end = chrono::high_resolution_clock::now();
        cout << "Strassen Sequential: " << chrono::duration<double, milli>(end - start).count() << " ms\n";

        start = chrono::high_resolution_clock::now();
        C_par = strassenMPI(A, B, n, rank, size);
        end = chrono::high_resolution_clock::now();
        cout << "Strassen MPI: " << chrono::duration<double, milli>(end - start).count() << " ms\n";

        cout << "Validation (Sequential): " << (validateMatrices(C_std, C_seq, n) ? "Passed" : "Failed") << endl;
        cout << "Validation (Parallel):   " << (validateMatrices(C_std, C_par, n) ? "Passed" : "Failed") << endl;

        for (int i = 1; i < size; ++i) {
            int newSize = -1;
            MPI_Send(&newSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }
    else {
        worker(rank);
    }

    MPI_Finalize();
    return 0;
}