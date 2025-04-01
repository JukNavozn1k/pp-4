#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

using namespace std;
using Matrix = vector<vector<int>>;

const int STRASSEN_THRESHOLD = 64;
const int TAG_TASK = 1;
const int TAG_RESULT = 2;

// Функция для генерации случайной матрицы
Matrix generateRandomMatrix(int n) {
    Matrix A(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = rand() % 10;
    return A;
}

// Стандартное умножение матриц
Matrix standardMultiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// Функции сложения и вычитания матриц
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

// Последовательный алгоритм Штрассена
Matrix strassenSequential(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD)
        return standardMultiply(A, B);

    int newSize = n / 2;
    Matrix A11(newSize, vector<int>(newSize)), A12(newSize, vector<int>(newSize)),
        A21(newSize, vector<int>(newSize)), A22(newSize, vector<int>(newSize));
    Matrix B11(newSize, vector<int>(newSize)), B12(newSize, vector<int>(newSize)),
        B21(newSize, vector<int>(newSize)), B22(newSize, vector<int>(newSize));

    // Разбиение матриц на 4 подматрицы
    for (int i = 0; i < newSize; i++)
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

    // Рекурсивные вызовы
    Matrix M1 = strassenSequential(add(A11, A22), add(B11, B22));
    Matrix M2 = strassenSequential(add(A21, A22), B11);
    Matrix M3 = strassenSequential(A11, subtract(B12, B22));
    Matrix M4 = strassenSequential(A22, subtract(B21, B11));
    Matrix M5 = strassenSequential(add(A11, A12), B22);
    Matrix M6 = strassenSequential(subtract(A21, A11), add(B11, B12));
    Matrix M7 = strassenSequential(subtract(A12, A22), add(B21, B22));

    // Сборка результирующей матрицы
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + newSize] = M3[i][j] + M5[i][j];
            C[i + newSize][j] = M2[i][j] + M4[i][j];
            C[i + newSize][j + newSize] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    return C;
}

// Вспомогательные функции для упаковки/распаковки матриц в одномерный вектор (для MPI)
vector<int> flattenMatrix(const Matrix& M) {
    int n = M.size();
    vector<int> flat;
    flat.reserve(n * n);
    for (const auto& row : M)
        for (int val : row)
            flat.push_back(val);
    return flat;
}

Matrix unflattenMatrix(const vector<int>& flat, int n) {
    Matrix M(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = flat[i * n + j];
    return M;
}

// Функция, выполняемая рабочим процессом: получение задачи, вычисление и отправка результата
void workerProcess() {
    int n;
    MPI_Recv(&n, 1, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    vector<int> flatA(n * n);
    MPI_Recv(flatA.data(), n * n, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    vector<int> flatB(n * n);
    MPI_Recv(flatB.data(), n * n, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    Matrix A = unflattenMatrix(flatA, n);
    Matrix B = unflattenMatrix(flatB, n);

    // Вычисляем произведение с использованием последовательного алгоритма Штрассена
    Matrix C = strassenSequential(A, B);
    vector<int> flatC = flattenMatrix(C);

    MPI_Send(flatC.data(), n * n, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
}

// Параллельная реализация алгоритма Штрассена с использованием MPI (параллелизация на первом уровне рекурсии)
Matrix strassenMPI(const Matrix& A, const Matrix& B, int depth = 0) {
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD)
        return standardMultiply(A, B);

    int newSize = n / 2;
    Matrix A11(newSize, vector<int>(newSize)), A12(newSize, vector<int>(newSize)),
        A21(newSize, vector<int>(newSize)), A22(newSize, vector<int>(newSize));
    Matrix B11(newSize, vector<int>(newSize)), B12(newSize, vector<int>(newSize)),
        B21(newSize, vector<int>(newSize)), B22(newSize, vector<int>(newSize));

    for (int i = 0; i < newSize; i++)
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

    // Если мы на верхнем уровне (depth==0) и процессов достаточно, выполняем параллелизацию
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    // Ожидаем минимум 7+1 процессов (мастер + 6 работников)
    if (depth == 0 && worldSize >= 7 + 1) {
        // Подготовка задач для 7 рекурсивных вызовов:
        // M1 = strassenMPI(add(A11, A22), add(B11, B22)) - вычисляем локально
        Matrix M1 = strassenMPI(add(A11, A22), add(B11, B22), depth + 1);
        // Остальные задачи (M2 ... M7) отправляем работникам (ранги 1..6)
        Matrix opA2 = add(A21, A22); Matrix opB2 = B11;            // M2
        Matrix opA3 = A11;         Matrix opB3 = subtract(B12, B22); // M3
        Matrix opA4 = A22;         Matrix opB4 = subtract(B21, B11); // M4
        Matrix opA5 = add(A11, A12); Matrix opB5 = B22;            // M5
        Matrix opA6 = subtract(A21, A11); Matrix opB6 = add(B11, B12); // M6
        Matrix opA7 = subtract(A12, A22); Matrix opB7 = add(B21, B22); // M7

        vector<pair<Matrix, Matrix>> tasks = {
            {opA2, opB2},
            {opA3, opB3},
            {opA4, opB4},
            {opA5, opB5},
            {opA6, opB6},
            {opA7, opB7}
        };

        // Отправляем задачи рабочим процессам: задачи нумеруются от 0 до 5, а рабочие процессы имеют ранги 1..6
        for (int i = 0; i < tasks.size(); i++) {
            int dest = i + 1;
            int subSize = tasks[i].first.size();
            MPI_Send(&subSize, 1, MPI_INT, dest, TAG_TASK, MPI_COMM_WORLD);
            vector<int> flatA = flattenMatrix(tasks[i].first);
            vector<int> flatB = flattenMatrix(tasks[i].second);
            MPI_Send(flatA.data(), subSize * subSize, MPI_INT, dest, TAG_TASK, MPI_COMM_WORLD);
            MPI_Send(flatB.data(), subSize * subSize, MPI_INT, dest, TAG_TASK, MPI_COMM_WORLD);
        }

        // Получаем результаты от рабочих процессов
        vector<Matrix> M(7);
        M[0] = M1; // M1 вычислено локально
        for (int i = 0; i < tasks.size(); i++) {
            int src = i + 1;
            int subSize = tasks[i].first.size();
            vector<int> flatC(subSize * subSize);
            MPI_Recv(flatC.data(), subSize * subSize, MPI_INT, src, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            M[i + 1] = unflattenMatrix(flatC, subSize);
        }

        // Сборка результирующей матрицы по формулам Штрассена:
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
    }
    else {
        // Если мы не на верхнем уровне или процессов недостаточно, выполняем рекурсию последовательно
        Matrix M1 = strassenMPI(add(A11, A22), add(B11, B22), depth + 1);
        Matrix M2 = strassenMPI(add(A21, A22), B11, depth + 1);
        Matrix M3 = strassenMPI(A11, subtract(B12, B22), depth + 1);
        Matrix M4 = strassenMPI(A22, subtract(B21, B11), depth + 1);
        Matrix M5 = strassenMPI(add(A11, A12), B22, depth + 1);
        Matrix M6 = strassenMPI(subtract(A21, A11), add(B11, B12), depth + 1);
        Matrix M7 = strassenMPI(subtract(A12, A22), add(B21, B22), depth + 1);

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
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Если процесс не мастер, переходим к обработке задач
    if (rank != 0) {
        workerProcess();
        MPI_Finalize();
        return 0;
    }

    // Мастер: генерация матриц и тестирование алгоритмов
    int n = 512; // размер матрицы (должен быть степенью двойки)
    Matrix A = generateRandomMatrix(n);
    Matrix B = generateRandomMatrix(n);

    // Стандартное умножение
    auto start_std = chrono::high_resolution_clock::now();
    Matrix C_std = standardMultiply(A, B);
    auto end_std = chrono::high_resolution_clock::now();
    double time_std = chrono::duration<double, milli>(end_std - start_std).count();

    // Последовательное Штрассен умножение
    auto start_seq = chrono::high_resolution_clock::now();
    Matrix C_seq = strassenSequential(A, B);
    auto end_seq = chrono::high_resolution_clock::now();
    double time_seq = chrono::duration<double, milli>(end_seq - start_seq).count();

    // Параллельное Штрассен умножение с использованием MPI
    auto start_mpi = chrono::high_resolution_clock::now();
    Matrix C_mpi = strassenMPI(A, B);
    auto end_mpi = chrono::high_resolution_clock::now();
    double time_mpi = chrono::duration<double, milli>(end_mpi - start_mpi).count();

    cout << "Standard multiply: " << time_std << " ms" << endl;
    cout << "Sequential Strassen: " << time_seq << " ms" << endl;
    cout << "MPI Parallel Strassen: " << time_mpi << " ms" << endl;

    MPI_Finalize();
    return 0;
}
