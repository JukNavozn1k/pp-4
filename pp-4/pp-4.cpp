#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cassert>

using namespace std;
using Matrix = vector<vector<int>>;

const int STRASSEN_THRESHOLD = 64;
const int PARALLEL_DEPTH_THRESHOLD = 2; // Глубина, до которой используем MPI-распараллеливание
const int TAG_RESULT = 100;

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

// Последовательный алгоритм Штрассена (без MPI)
Matrix strassenSequential(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD)
        return standardMultiply(A, B);

    int newSize = n / 2;
    // Инициализация подматриц
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

// Упаковка матрицы в одномерный вектор (для передачи по MPI)
vector<int> flattenMatrix(const Matrix& M) {
    int n = M.size();
    vector<int> flat;
    flat.reserve(n * n);
    for (const auto& row : M)
        for (int val : row)
            flat.push_back(val);
    return flat;
}

// Восстановление матрицы из одномерного вектора
Matrix unflattenMatrix(const vector<int>& flat, int n) {
    Matrix M(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = flat[i * n + j];
    return M;
}

// Рекурсивная функция MPI-Штрассена с глубоким распараллеливанием до заданной глубины
Matrix strassenMPIDeep(const Matrix& A, const Matrix& B, MPI_Comm comm, int depth) {
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD)
        return standardMultiply(A, B);

    // Если достигли предельной глубины параллелизма – выполняем последовательный алгоритм
    if (depth >= PARALLEL_DEPTH_THRESHOLD)
        return strassenSequential(A, B);

    int newSize = n / 2;
    // Разбиение на подматрицы
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

    // Подготовка семи задач для рекурсивных вызовов:
    // M1 = (A11+A22) * (B11+B22)
    // M2 = (A21+A22) * B11
    // M3 = A11 * (B12-B22)
    // M4 = A22 * (B21-B11)
    // M5 = (A11+A12) * B22
    // M6 = (A21-A11) * (B11+B12)
    // M7 = (A12-A22) * (B21+B22)
    vector<Matrix> taskA(7), taskB(7);
    taskA[0] = add(A11, A22);      taskB[0] = add(B11, B22);
    taskA[1] = add(A21, A22);      taskB[1] = B11;
    taskA[2] = A11;              taskB[2] = subtract(B12, B22);
    taskA[3] = A22;              taskB[3] = subtract(B21, B11);
    taskA[4] = add(A11, A12);      taskB[4] = B22;
    taskA[5] = subtract(A21, A11); taskB[5] = add(B11, B12);
    taskA[6] = subtract(A12, A22); taskB[6] = add(B21, B22);

    // Получаем информацию по коммуникатору
    int worldSize, worldRank;
    MPI_Comm_size(comm, &worldSize);
    MPI_Comm_rank(comm, &worldRank);

    // Для глубокого распараллеливания мы распределяем 7 задач между процессами в данном коммуникаторе.
    // Если процессов достаточно (worldSize >= 7), каждый процесс вычисляет задачу, определяемую по остатку от деления его ранга.
    // Если процессов меньше – выполняем последовательный вариант.
    vector<Matrix> M_tasks(7);
    if (worldSize >= 7) {
        int myTask = worldRank % 7; // каждому процессу назначается одна из 7 задач
        // Создаём подкоммуникатор для группы процессов, вычисляющих одну и ту же задачу
        MPI_Comm subcomm;
        MPI_Comm_split(comm, myTask, worldRank, &subcomm);
        int subRank;
        MPI_Comm_rank(subcomm, &subRank);
        Matrix localResult;
        // Пусть только лидер подгруппы (subRank==0) вычисляет рекурсивный вызов,
        // а затем результат транслируется всем участникам подгруппы.
        if (subRank == 0) {
            localResult = strassenMPIDeep(taskA[myTask], taskB[myTask], subcomm, depth + 1);
        }
        // Определяем размер подматрицы результата (newSize, т.к. taskA[*] имеют размер newSize)
        int subN = taskA[myTask].size();
        vector<int> flatRes(subN * subN);
        if (subRank == 0) {
            flatRes = flattenMatrix(localResult);
        }
        MPI_Bcast(flatRes.data(), subN * subN, MPI_INT, 0, subcomm);
        // Все процессы в подгруппе теперь имеют один и тот же результат
        localResult = unflattenMatrix(flatRes, subN);
        M_tasks[myTask] = localResult;
        MPI_Comm_free(&subcomm);

        // Теперь глобально собираем результаты. Пусть для каждой задачи лидер – процесс с рангом равным номеру задачи (если такой есть).
        // Если глобальный лидер не является лидером подгруппы, то его группа уже вычислила результат.
        if (worldRank < 7 && worldRank != (worldRank % 7)) {
            // Это условие маловероятно, т.к. если worldRank < 7, то worldRank % 7 == worldRank.
        }
        // Глобальный процесс с рангом 0 собирает результаты от лидеров групп для всех задач.
        vector<Matrix> gatheredTasks(7);
        if (worldRank == 0) {
            // Для задачи 0, результат уже есть локально
            gatheredTasks[0] = M_tasks[0];
            for (int t = 1; t < 7; t++) {
                int subN = taskA[t].size();
                vector<int> flatTask(subN * subN);
                MPI_Recv(flatTask.data(), subN * subN, MPI_INT, t, TAG_RESULT, comm, MPI_STATUS_IGNORE);
                gatheredTasks[t] = unflattenMatrix(flatTask, subN);
            }
        }
        else {
            // Если процесс является лидером подгруппы, то если его глобальный ранг совпадает с номером задачи, отправляем результат глобальному процессу 0
            if (worldRank < 7) {
                int subN = taskA[worldRank].size();
                vector<int> flatTask = flattenMatrix(M_tasks[worldRank]);
                MPI_Send(flatTask.data(), subN * subN, MPI_INT, 0, TAG_RESULT, comm);
            }
        }

        // Глобальный процесс 0 собирает все результаты и собирает итоговую матрицу
        Matrix C;
        if (worldRank == 0) {
            C = Matrix(n, vector<int>(n));
            for (int i = 0; i < newSize; i++) {
                for (int j = 0; j < newSize; j++) {
                    C[i][j] = gatheredTasks[0][i][j] + gatheredTasks[3][i][j] - gatheredTasks[4][i][j] + gatheredTasks[6][i][j];
                    C[i][j + newSize] = gatheredTasks[2][i][j] + gatheredTasks[4][i][j];
                    C[i + newSize][j] = gatheredTasks[1][i][j] + gatheredTasks[3][i][j];
                    C[i + newSize][j + newSize] = gatheredTasks[0][i][j] - gatheredTasks[1][i][j] + gatheredTasks[2][i][j] + gatheredTasks[5][i][j];
                }
            }
            // После сборки результата, рассылаем его всем процессам
            vector<int> flatC = flattenMatrix(C);
            MPI_Bcast(flatC.data(), n * n, MPI_INT, 0, comm);
        }
        else {
            vector<int> flatC(n * n);
            MPI_Bcast(flatC.data(), n * n, MPI_INT, 0, comm);
            C = unflattenMatrix(flatC, n);
        }
        return C;
    }
    else {
        // Если процессов недостаточно для параллелизации на этом уровне, переходим к последовательному Штрассену
        return strassenSequential(A, B);
    }
}

// Функция сравнения матриц (возвращает true, если матрицы совпадают)
bool compareMatrices(const Matrix& A, const Matrix& B) {
    if (A.size() != B.size()) return false;
    int n = A.size();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (A[i][j] != B[i][j])
                return false;
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    // Пусть глобально процесс с рангом 0 будет мастером
    Matrix A, B;
    int n = 512; // Размер матрицы (степень двойки)
    if (worldRank == 0) {
        A = generateRandomMatrix(n);
        B = generateRandomMatrix(n);
    }

    // Рассылка размеров и матриц всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (worldRank != 0) {
        // Остальные процессы инициализируют матрицы нужного размера
        A = Matrix(n, vector<int>(n));
        B = Matrix(n, vector<int>(n));
    }
    // Преобразуем матрицы в одномерные векторы для рассылки
    vector<int> flatA, flatB;
    if (worldRank == 0) {
        flatA = flattenMatrix(A);
        flatB = flattenMatrix(B);
    }
    else {
        flatA.resize(n * n);
        flatB.resize(n * n);
    }
    MPI_Bcast(flatA.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(flatB.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);
    if (worldRank != 0) {
        A = unflattenMatrix(flatA, n);
        B = unflattenMatrix(flatB, n);
    }

    // На мастере выполняем вычисления для сравнения
    Matrix C_std, C_seq, C_mpi;
    double time_std = 0.0, time_seq = 0.0, time_mpi = 0.0;
    if (worldRank == 0) {
        auto start = chrono::high_resolution_clock::now();
        C_std = standardMultiply(A, B);
        auto end = chrono::high_resolution_clock::now();
        time_std = chrono::duration<double, milli>(end - start).count();

        start = chrono::high_resolution_clock::now();
        C_seq = strassenSequential(A, B);
        end = chrono::high_resolution_clock::now();
        time_seq = chrono::duration<double, milli>(end - start).count();
    }

    // Все процессы вызывают MPI-версию (глубокий MPI Штрассен)
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_mpi = chrono::high_resolution_clock::now();
    C_mpi = strassenMPIDeep(A, B, MPI_COMM_WORLD, 0);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_mpi = chrono::high_resolution_clock::now();
    time_mpi = chrono::duration<double, milli>(end_mpi - start_mpi).count();

    // Мастер выводит результаты и проверяет корректность
    if (worldRank == 0) {
        cout << "Standard multiply: " << time_std << " ms" << endl;
        cout << "Sequential Strassen: " << time_seq << " ms" << endl;
        cout << "MPI Parallel (Deep) Strassen: " << time_mpi << " ms" << endl;

        bool ok1 = compareMatrices(C_std, C_seq);
        bool ok2 = compareMatrices(C_std, C_mpi);
        cout << "Standard vs Sequential Strassen: " << (ok1 ? "OK" : "Mismatch") << endl;
        cout << "Standard vs MPI Strassen: " << (ok2 ? "OK" : "Mismatch") << endl;
    }

    MPI_Finalize();
    return 0;
}
