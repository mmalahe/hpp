/** @file testFFTWParallel.cpp
* @author Michael Malahe
* @brief Tests for FFTW parallel functions
*/

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mpi.h>
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <hpp/spectralUtils.h>

const double closeEnough = 10*std::numeric_limits<double>::epsilon();

/**
 * @brief Computes a 2D real FFT, followed by IFFT and checks that the
 * original array has been returned.
 */
void testFFTW2DReal(MPI_Comm comm) 
{ 
    // Global problem size
    ptrdiff_t N0Real = 128;
    ptrdiff_t N1Real = 128;
    ptrdiff_t NReal = N0Real*N1Real;
    ptrdiff_t N0Complex = N0Real;
    ptrdiff_t N1Complex = N1Real/2+1;
    ptrdiff_t N1RealPadded = 2*N1Complex;
    
    // Local problem size
    ptrdiff_t localN0;
    ptrdiff_t local0Start;
    
    // Allocate values in and out
    ptrdiff_t nLocalComplex = fftw_mpi_local_size_2d(N0Complex, N1Complex, comm, &localN0, &local0Start);
    ptrdiff_t nLocalReal = 2*nLocalComplex;
    double *in = fftw_alloc_real(nLocalReal);
    fftw_complex *out = fftw_alloc_complex(nLocalComplex);
    double *backin = fftw_alloc_real(nLocalReal);
    
    // Input values
    for (int i=0; i<localN0; i++) {
        for (int j=0; j<N1Real; j++) {
            in[hpp::fftwFlat2(i,j,localN0,N1RealPadded)] = std::sin(2*(i+local0Start)*M_PI/N0Real) + std::cos(2*j*M_PI/N1Real);
        }
    }
    
    // FFT
    
    // Execution plan
    fftw_plan forward = fftw_mpi_plan_dft_r2c_2d(N0Real, N1Real, in, out, comm, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(forward);
    
    // Scale correctly
    for (int i=0; i<nLocalComplex; i++) {
        out[i][0] /= NReal;
        out[i][1] /= NReal;
    }
    
    // IFFT
    fftw_plan backward = fftw_mpi_plan_dft_c2r_2d(N0Real, N1Real, out, backin, comm, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(backward);
    
    // Compare
    for (int i=0; i<localN0; i++) {
        for (int j=0; j<N1Real; j++) {
            int idx = hpp::fftwFlat2(i,j,localN0,N1RealPadded);
            if (std::abs(in[idx]-backin[idx]) > closeEnough) throw std::runtime_error("Real part didn't match.");
        }
    }
    
    // Free
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(in);
    fftw_free(out);
    fftw_free(backin);
}

/**
 * @brief Computes an ND real FFT, followed by IFFT and checks that the
 * original array has been returned.
 */
void testFFTW4DReal(MPI_Comm comm) 
{ 
    // Rank
    unsigned int rank = 4;
    
    // Global problem size
    int N = 16;
    std::vector<ptrdiff_t> realDims(rank, N);    
    std::vector<ptrdiff_t> complexDims(rank, N);
    complexDims.back() = N/2+1;
    int NReal = 1;
    for (auto dim : realDims) {
        NReal *= dim;
    }
    int NComplex = 1;
    for (auto dim : complexDims) {
        NComplex *= dim;
    }
    std::vector<ptrdiff_t> realDimsPadded = realDims;
    realDimsPadded.back() = complexDims.back()*2;
    
    // Local problem size
    ptrdiff_t localN0;
    ptrdiff_t local0Start;
    
    // Allocate values in and out
    ptrdiff_t nLocalComplex = fftw_mpi_local_size(rank, complexDims.data(), comm, &localN0, &local0Start);
    ptrdiff_t nLocalRealPadded = 2*nLocalComplex;
    double *in = fftw_alloc_real(nLocalRealPadded);
    fftw_complex *out = fftw_alloc_complex(nLocalComplex);
    double *backin = fftw_alloc_real(nLocalRealPadded);
    
    // Local dimensions
    std::vector<ptrdiff_t> realDimsPaddedLocal = realDimsPadded;
    realDimsPaddedLocal[0] = localN0;
    std::vector<ptrdiff_t> realDimsLocal = realDims;
    realDimsLocal[0] = localN0;
    
    // Input values
    for (unsigned int i=0; i<realDimsLocal[0]; i++) {
        for (unsigned int j=0; j<realDimsLocal[1]; j++) {
            for (unsigned int k=0; k<realDimsLocal[2]; k++) {
                for (unsigned int l=0; l<realDimsLocal[3]; l++) {
                    int idx = hpp::fftwFlat4(i,j,k,l,realDimsPaddedLocal);
                    in[idx] = std::sin(2*(i+local0Start)*M_PI/realDims[0]) + std::cos(2*j*M_PI/realDims[1]) + k + l;
                }
            }
        }
    }
    
    // FFT
    
    // Execution plan
    fftw_plan forward = fftw_mpi_plan_dft_r2c(rank, realDims.data(), in, out, comm, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(forward);
    
    // Scale correctly
    for (int i=0; i<nLocalComplex; i++) {
        out[i][0] /= NReal;
        out[i][1] /= NReal;
    }
    
    // IFFT
    fftw_plan backward = fftw_mpi_plan_dft_c2r(rank, realDims.data(), out, backin, comm, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(backward);
    
    // Compare
    for (int i=0; i<realDimsLocal[0]; i++) {
        for (int j=0; j<realDimsLocal[1]; j++) {
            for (int k=0; k<realDimsLocal[2]; k++) {
                for (int l=0; l<realDimsLocal[3]; l++) {
                    int idx = hpp::fftwFlat4(i,j,k,l,realDimsPaddedLocal);
                    double approxCondNumberScaling = std::max(std::abs(in[idx]), 1.0)*N;
                    double diff = std::abs(in[idx]-backin[idx]);
                    double threshold = closeEnough*approxCondNumberScaling;
                    if (diff > threshold) {
                        std::cout << in[idx] << " " << backin[idx] << " " << diff << " " << threshold << std::endl;
                        throw std::runtime_error("Real part didn't match.");
                    }
                }
            }
        }
    }
    
    // Free
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(in);
    fftw_free(out);
    fftw_free(backin);
}

/**
 * @brief Computes a 2D real FFT, followed by IFFT and checks that the
 * original array has been returned.
 * @details Slightly faster transposed version
 */
void testFFTW2DRealTransposed(MPI_Comm comm) 
{ 
    // Global problem size
    ptrdiff_t N0Real = 128;
    ptrdiff_t N1Real = 128;
    ptrdiff_t NReal = N0Real*N1Real;
    ptrdiff_t N0Complex = N0Real;
    ptrdiff_t N1Complex = N1Real/2+1;
    ptrdiff_t N1RealPadded = 2*N1Complex;
    
    // Local problem size
    ptrdiff_t localN0;
    ptrdiff_t local0Start;
    
    // Local problem size transposed
    ptrdiff_t localN1;
    ptrdiff_t local1Start;
    
    // Allocate values in and out
    ptrdiff_t nLocalComplex = fftw_mpi_local_size_2d_transposed(N0Complex, N1Complex, comm, &localN0, &local0Start, &localN1, &local1Start);
    ptrdiff_t nLocalReal = 2*nLocalComplex;
    double *in = fftw_alloc_real(nLocalReal);
    fftw_complex *out = fftw_alloc_complex(nLocalComplex);
    double *backin = fftw_alloc_real(nLocalReal);
    
    // Input values
    for (int i=0; i<localN0; i++) {
        for (int j=0; j<N1Real; j++) {
            in[hpp::fftwFlat2(i,j,localN0,N1RealPadded)] = std::sin(2*(i+local0Start)*M_PI/N0Real) + std::cos(2*j*M_PI/N1Real);
        }
    }
    
    // FFT
    
    // Execution plan
    fftw_plan forward = fftw_mpi_plan_dft_r2c_2d(N0Real, N1Real, in, out, comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
    
    // Execute
    fftw_execute(forward);
    
    // Scale correctly
    for (int i=0; i<nLocalComplex; i++) {
        out[i][0] /= NReal;
        out[i][1] /= NReal;
    }
    
    // IFFT
    fftw_plan backward = fftw_mpi_plan_dft_c2r_2d(N0Real, N1Real, out, backin, comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
    
    // Execute
    fftw_execute(backward);
    
    // Compare
    for (int i=0; i<localN0; i++) {
        for (int j=0; j<N1Real; j++) {
            int idx = hpp::fftwFlat2(i,j,localN0,N1RealPadded);
            if (std::abs(in[idx]-backin[idx]) > closeEnough) throw std::runtime_error("Real part didn't match.");
        }
    }
    
    // Free
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(in);
    fftw_free(out);
    fftw_free(backin);
}

int main(int argc, char *argv[]) {
    // Init
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    fftw_mpi_init();
    
    // Parallel tests
    testFFTW2DReal(comm);
    testFFTW2DRealTransposed(comm);
    testFFTW4DReal(comm);
    
    // Finalize
    fftw_mpi_cleanup();
    MPI_Finalize();
    
    // Return
    return 0;
}