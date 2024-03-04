export OMP_NUM_THREADS=1
#python -c "import fenics" # this does not work
mpirun -n 1 python -c "import fenics; print(fenics.MPI.comm_world.rank)"

