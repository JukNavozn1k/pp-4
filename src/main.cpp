#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <mpi.h>

using namespace std;
using Matrix = vector<vector<int>>;

const int STRASSEN_THRESHOLD = 64;
const int TERMINATE_TAG = -1;
const int WORK_TAG = 1;

// Функция сравнения матриц
bool compareMatrices(const Matrix& A, const Matrix& B, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (A[i][j] != B[i][j]) {
                cout << "Mismatch at (" << i << "," << j << "): "
                     << A[i][j] << " vs " << B[i][j] << endl;
                return false;
            }
        }
    }
    return true;
}

Matrix generateRandomMatrix(int n) {
    Matrix A(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = rand() % 10;
    return A;
}

// Остальные функции остаются без изменений (add, subtract, standardMultiply, sendMatrix, receiveMatrix, strassen и т.д.)

int nextPowerOfTwo(int n) {
    return pow(2, ceil(log2(n)));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            cerr << "Usage: " << argv[0] << " <matrix_size>" << endl;
        MPI_Finalize();
        return 1;
    }

    int original_n = atoi(argv[1]);
    int n = nextPowerOfTwo(original_n);
    
    Matrix A, B, C_std, C_seq, C_par;
    double start, end;

    if (rank == 0) {
        // Генерация матриц исходного размера
        Matrix A_original = generateRandomMatrix(original_n);
        Matrix B_original = generateRandomMatrix(original_n);

        // Дополнение нулями до степени двойки
        A = Matrix(n, vector<int>(n, 0));
        B = Matrix(n, vector<int>(n, 0));
        for (int i = 0; i < original_n; i++) {
            for (int j = 0; j < original_n; j++) {
                A[i][j] = A_original[i][j];
                B[i][j] = B_original[i][j];
            }
        }

        // Стандартное умножение
        start = MPI_Wtime();
        C_std = standardMultiply(A, B);
        end = MPI_Wtime();
        cout << "Standard multiply: " << (end - start) * 1000 << " ms" << endl;

        // Последовательный Штрассен
        start = MPI_Wtime();
        C_seq = strassen(A, B, 0, 1, 0);
        end = MPI_Wtime();
        cout << "Sequential Strassen: " << (end - start) * 1000 << " ms" << endl;

        // Проверка последовательной версии
        Matrix C_std_cropped(original_n, vector<int>(original_n));
        Matrix C_seq_cropped(original_n, vector<int>(original_n));
        for (int i = 0; i < original_n; i++) {
            for (int j = 0; j < original_n; j++) {
                C_std_cropped[i][j] = C_std[i][j];
                C_seq_cropped[i][j] = C_seq[i][j];
            }
        }
        
        cout << "Verifying Sequential Strassen... ";
        if (compareMatrices(C_std_cropped, C_seq_cropped, original_n)) {
            cout << "OK" << endl;
        } else {
            cout << "FAIL" << endl;
        }

        // Параллельный Штрассен
        start = MPI_Wtime();
    }

    if (rank == 0) {
        C_par = strassen(A, B, 0, size, 0);
        end = MPI_Wtime();
        cout << "Parallel Strassen: " << (end - start) * 1000 << " ms" << endl;

        // Проверка параллельной версии
        Matrix C_par_cropped(original_n, vector<int>(original_n));
        for (int i = 0; i < original_n; i++) {
            for (int j = 0; j < original_n; j++) {
                C_par_cropped[i][j] = C_par[i][j];
            }
        }
        
        cout << "Verifying Parallel Strassen... ";
        if (compareMatrices(C_std_cropped, C_par_cropped, original_n)) {
            cout << "OK" << endl;
        } else {
            cout << "FAIL" << endl;
        }

    } else {
        strassenWorker(rank, 0);
    }

    MPI_Finalize();
    return 0;
}