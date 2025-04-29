# Parallel Matrix Multiplication with Strassen's Algorithm

This project implements parallel matrix multiplication using Strassen's algorithm with MPI for distributed computation.

## Features

- Parallel matrix multiplication using MPI.
- Strassen's algorithm for efficient matrix multiplication.
- Support for both sequential and parallel execution.
- Comparison of standard, sequential Strassen, and MPI-based Strassen implementations.

## Requirements

- C++ compiler with MPI support (e.g., `mpic++`).
- MPI runtime (e.g., OpenMPI or MPICH).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/pp-4.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pp-4
   ```
3. Build the project using `mpic++`:
   ```bash
   mpic++ -o pp-4 pp-4/pp-4.cpp -std=c++17
   ```

## Usage

Run the application with MPI:
```bash
mpirun -np <number_of_processes> ./pp-4
```

Replace `<number_of_processes>` with the desired number of MPI processes.

## How It Works

1. **Matrix Generation**: Random matrices are generated for multiplication.
2. **Standard Multiplication**: A baseline implementation for comparison.
3. **Strassen's Algorithm**: A sequential implementation of Strassen's algorithm.
4. **MPI Parallelization**: Deep parallelization of Strassen's algorithm using MPI.

## Example Output

The program compares the performance of different implementations:
```
Standard multiply: 1234.56 ms
Sequential Strassen: 789.01 ms
MPI Parallel (Deep) Strassen: 456.78 ms
Standard vs Sequential Strassen: OK
Standard vs MPI Strassen: OK
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).


### For vscode (include path on WSL ubuntu)
```
${workspaceFolder}/**
"/usr/include"
/usr/lib/x86_64-linux-gnu/openmpi/include
```