#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <mpi.h>

using namespace std;
using Matrix = vector<vector<int>>;

const int STRASSEN_THRESHOLD = 64;

Matrix generateRandomMatrix(int n) {
    Matrix A(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = rand() % 10;
    return A;
}

Matrix standardMultiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

Matrix add(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix subtract(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

Matrix strassenSequential(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD) return standardMultiply(A, B);

    int newSize = n / 2;
    Matrix A11(newSize, vector<int>(newSize)), A12(newSize, vector<int>(newSize)),
           A21(newSize, vector<int>(newSize)), A22(newSize, vector<int>(newSize));
    Matrix B11(newSize, vector<int>(newSize)), B12(newSize, vector<int>(newSize)),
           B21(newSize, vector<int>(newSize)), B22(newSize, vector<int>(newSize));

    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

    Matrix M1 = strassenSequential(add(A11, A22), add(B11, B22));
    Matrix M2 = strassenSequential(add(A21, A22), B11);
    Matrix M3 = strassenSequential(A11, subtract(B12, B22));
    Matrix M4 = strassenSequential(A22, subtract(B21, B11));
    Matrix M5 = strassenSequential(add(A11, A12), B22);
    Matrix M6 = strassenSequential(subtract(A21, A11), add(B11, B12));
    Matrix M7 = strassenSequential(subtract(A12, A22), add(B21, B22));

    Matrix C(n, vector<int>(n));
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + newSize] = M3[i][j] + M5[i][j];
            C[i + newSize][j] = M2[i][j] + M4[i][j];
            C[i + newSize][j + newSize] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }
    return C;
}

void matrixToArray(const Matrix& M, int* arr) {
    int k = 0;
    for (const auto& row : M)
        for (int val : row)
            arr[k++] = val;
}

Matrix arrayToMatrix(const int* arr, int size) {
    Matrix M(size, vector<int>(size));
    int k = 0;
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            M[i][j] = arr[k++];
    return M;
}

Matrix strassenMPI(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty()) return Matrix();
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD) return standardMultiply(A, B);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0) {
        return strassenSequential(A, B);
    }

    int newSize = n / 2;
    Matrix A11(newSize, vector<int>(newSize)), A12(newSize, vector<int>(newSize)),
           A21(newSize, vector<int>(newSize)), A22(newSize, vector<int>(newSize));
    Matrix B11(newSize, vector<int>(newSize)), B12(newSize, vector<int>(newSize)),
           B21(newSize, vector<int>(newSize)), B22(newSize, vector<int>(newSize));

    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

    if (size >= 8) {
        vector<Matrix> A_parts = {add(A11, A22), add(A21, A22), A11, A22, add(A11, A12), subtract(A21, A11), subtract(A12, A22)};
        vector<Matrix> B_parts = {add(B11, B22), B11, subtract(B12, B22), subtract(B21, B11), B22, add(B11, B12), add(B21, B22)};

        for (int worker = 1; worker <= 7; worker++) {
            int idx = worker - 1;
            int part_size = newSize;
            vector<int> A_buffer(part_size * part_size);
            vector<int> B_buffer(part_size * part_size);
            matrixToArray(A_parts[idx], A_buffer.data());
            matrixToArray(B_parts[idx], B_buffer.data());

            MPI_Send(&part_size, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            MPI_Send(A_buffer.data(), part_size * part_size, MPI_INT, worker, 0, MPI_COMM_WORLD);
            MPI_Send(B_buffer.data(), part_size * part_size, MPI_INT, worker, 0, MPI_COMM_WORLD);
        }

        vector<Matrix> M(7);
        for (int worker = 1; worker <= 7; worker++) {
            int part_size;
            MPI_Recv(&part_size, 1, MPI_INT, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            vector<int> buffer(part_size * part_size);
            MPI_Recv(buffer.data(), part_size * part_size, MPI_INT, worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            M[worker-1] = arrayToMatrix(buffer.data(), part_size);
        }

        Matrix C(n, vector<int>(n));
        for (int i = 0; i < newSize; i++) {
            for (int j = 0; j < newSize; j++) {
                C[i][j] = M[0][i][j] + M[3][i][j] - M[4][i][j] + M[6][i][j];
                C[i][j + newSize] = M[2][i][j] + M[4][i][j];
                C[i + newSize][j] = M[1][i][j] + M[3][i][j];
                C[i + newSize][j + newSize] = M[0][i][j] - M[1][i][j] + M[2][i][j] + M[5][i][j];
            }
        }
        return C;
    } else {
        return strassenSequential(A, B);
    }
}

bool areMatricesEqual(const Matrix& A, const Matrix& B) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); i++) {
        if (A[i].size() != B[i].size()) return false;
        for (size_t j = 0; j < A[i].size(); j++) {
            if (A[i][j] != B[i][j]) return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 512;
    Matrix A, B, C_std, C_seq, C_par;
    double time_std = 0.0, time_seq = 0.0, time_par = 0.0;

    if (rank == 0) {
        A = generateRandomMatrix(n);
        B = generateRandomMatrix(n);

        auto start_std = chrono::high_resolution_clock::now();
        C_std = standardMultiply(A, B);
        auto end_std = chrono::high_resolution_clock::now();
        time_std = chrono::duration<double, milli>(end_std - start_std).count();

        auto start_seq = chrono::high_resolution_clock::now();
        C_seq = strassenSequential(A, B);
        auto end_seq = chrono::high_resolution_clock::now();
        time_seq = chrono::duration<double, milli>(end_seq - start_seq).count();

        auto start_par = chrono::high_resolution_clock::now();
        C_par = strassenMPI(A, B);
        auto end_par = chrono::high_resolution_clock::now();
        time_par = chrono::duration<double, milli>(end_par - start_par).count();

        cout << "Standard multiply: " << time_std << " ms\n";
        cout << "Sequential Strassen: " << time_seq << " ms\n";
        cout << "Parallel Strassen (MPI): " << time_par << " ms\n";

        bool valid_seq = areMatricesEqual(C_std, C_seq);
        bool valid_par = areMatricesEqual(C_std, C_par);

        cout << "Validation sequential Strassen: " << (valid_seq ? "Success" : "Failure") << endl;
        cout << "Validation parallel Strassen: " << (valid_par ? "Success" : "Failure") << endl;
    } else {
        strassenMPI(Matrix(), Matrix());
    }

    MPI_Finalize();
    return 0;
}