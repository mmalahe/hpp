#include <hpp/mpiUtils.h>

namespace hpp
{

bool MPIAllTrue(bool condition, MPI_Comm comm) {
    int localCondition = (int)condition;
    int globalCondition;
    MPI_Allreduce(&localCondition, &globalCondition, 1, MPI_INT, MPI_LAND, comm);
    return (bool)globalCondition;
}

}//END NAMESPACE HPP