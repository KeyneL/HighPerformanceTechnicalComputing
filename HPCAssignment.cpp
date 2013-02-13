#include "mpi.h"
#include <cmath>
#include <iostream>
#include <fstream>

const static int N = 50;
const static int MAXITER = 100;

int main(int argc, char *argv[]) {
	 int numprocs, rank;
	 int tag = 1;
	 double rmatrix[N+1][N+1];
	 double pi = atan(1.0)*4.0;
	 MPI_Init(&argc, &argv);
	 MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	 MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	 MPI_Status stat;

	for(int i = 0; i < N+1; ++i)
		for(int j = 0; j < N+1; ++j){
			rmatrix[i][j] = 0.0;
			}

  	for(int i = 0; i < N+1; ++i){
		rmatrix[i][0] = pow(sin(pi * ((double)i/N)), 2);
	}

	double half = numprocs/2;
	int *startx = new int [numprocs];
	int *endx = new int [numprocs];
	int *starty = new int [numprocs];
	int *endy = new int [numprocs];

	for(int i = 0; i < numprocs; ++i){
		if(i < half){
			startx[i] = 1;
			endx[i] = N/2 + 1;
			starty[i] = i*(N/half);
			if(i == 0) starty[i] += 1;
			endy[i] = (i+1)*(N/half);
		}
		else{
			startx[i] = N/2 + 1;
			endx[i] = N;
			starty[i] = (i-half) * (N/half);
			if(i-half == 0) starty[i] += 1;
			endy[i] = (i-half+1) * (N/half);
		}
	}

	double** newmatrix = new double*[endy[rank] - starty[rank]];
	for(int i = 0; i < endy[rank]-starty[rank]; ++i)
		newmatrix[i] = new double[endx[rank] - startx[rank]];

	double starttime = MPI_Wtime();

	for(int i = 0; i < MAXITER; ++i){
		for(int x = startx[rank]; x < endx[rank]; ++x)
			for(int y = starty[rank]; y < endy[rank]; ++y)
				newmatrix[y - starty[rank]][x - startx[rank]] = (rmatrix[y][x-1] + rmatrix[y][x+1] + rmatrix[y-1][x] + rmatrix[y+1][x])/4;

		for(int x = startx[rank]; x < endx[rank]; ++x)
			for(int y = starty[rank]; y < endy[rank]; ++y)
				rmatrix[y][x] = newmatrix[y-starty[rank]][x-startx[rank]];

		if(i != MAXITER -1){
			int index;
			double* botarray;
			int dest = rank+1;
			int source = rank-1;
			if(rank == half - 1 || rank == numprocs-1)
				dest = MPI_PROC_NULL;
			if(rank == 0 || rank == half)
				source = MPI_PROC_NULL;

			botarray = new double[endx[rank]-startx[rank]];

			MPI_Sendrecv(newmatrix[endy[rank] - starty[rank] - 1], endx[rank]-startx[rank], MPI_DOUBLE, dest, tag,
				botarray, endx[rank]-startx[rank], MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &stat);

			index = 0;
			if(source != MPI_PROC_NULL)
				for(int x = startx[rank]; x < endx[rank]; ++x, ++index)
					rmatrix[starty[rank]-1][x] = botarray[index];

			delete[] botarray;

			double* toparray;
			dest = rank-1;
			source = rank+1;
			if(rank == half - 1 || rank == numprocs-1)
				source = MPI_PROC_NULL;
	
			if(rank == 0 ||rank == half)
				dest = MPI_PROC_NULL;

			toparray = new double[endx[rank]-startx[rank]];

			MPI_Sendrecv(newmatrix[0], endx[rank] - startx[rank], MPI_DOUBLE, dest, tag,
				toparray, endx[rank] - startx[rank], MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &stat);

			index = 0;
			if(source != MPI_PROC_NULL)
				for(int x = startx[rank]; x < endx[rank]; ++x, ++index)
					rmatrix[endy[rank]][x] = toparray[index];

			delete[] toparray;

			double *sidetoreceive = new double[endy[rank] - starty[rank]];
			double *sidetosend = new double[endy[rank]-starty[rank]];
			if(rank < half){
				source = dest = rank + half;
				index = 0;
				for(int y = starty[rank]; y < endy[rank]; ++y, ++index)
					sidetosend[index] = newmatrix[y - starty[rank]][endx[rank] - startx[rank] - 1];
			}
			
			else{
				source = dest = rank - half;
				index = 0;
				for(int y = starty[rank]; y < endy[rank]; ++y, ++index)
					sidetosend[index] = newmatrix[y - starty[rank]][0];
			}

			MPI_Sendrecv(sidetosend, endy[rank]-starty[rank], MPI_DOUBLE, dest, tag, sidetoreceive,
				endy[rank] - starty[rank], MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &stat);

			index = 0;
			for(int y = starty[rank]; y < endy[rank]; ++y, ++index){
				if(rank > source) rmatrix[y][startx[rank]-1] = sidetoreceive[index];
				else rmatrix[y][endx[rank]] = sidetoreceive[index];
			}

			delete[] sidetosend;
			delete[] sidetoreceive;
		}
	}

	if(rank != 0){
		for(int y = starty[rank]; y < endy[rank]; ++y){
			double *datatoreduce = new double[endx[rank] - startx[rank]];
			int index = 0;
				for(int x = startx[rank]; x < endx[rank]; ++x){
					datatoreduce[index] = rmatrix[y][x];
					++index;
				}

			MPI_Send(datatoreduce, endx[rank] - startx[rank] , MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
			delete[] datatoreduce;
		}
	}					
	
	else{
		double *datareceived;
		for(int i = 1; i < numprocs; ++i){
			for(int j = starty[i]; j < endy[i]; ++j){
				datareceived = new double[endx[i] - startx[i]];
				MPI_Recv(datareceived, endx[i] - startx[i] , MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &stat);
				int index = 0;
				for(int x = startx[rank]; x < endx[rank]; ++x){
					rmatrix[j][x] = datareceived[index];
					++index;
				}
				delete[] datareceived;
			}
		}

		std::ofstream file;
		file.open("Result.txt");
		for(int i = 0; i < N; ++i){
			for(int j = 0; j < N; ++j)
				file << std::fixed << rmatrix[i][j] << " ";
			file << "\n";
		}
		file.close();
		double endtime = MPI_Wtime();
		std::cout << std::fixed << "Time to perform the calculation = " << starttime - endtime << " seconds" << std::endl;
	}

    MPI_Finalize();
	return 0;
}