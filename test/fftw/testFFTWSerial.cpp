/** @file testFFTWSerial.cpp
* @author Michael Malahe
* @brief Tests for FFTW serial functions
*/

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <fftw3.h>
#include <hpp/spectralUtils.h>

const double closeEnough = 100*std::numeric_limits<double>::epsilon();

/**
 * @brief Computes a 1D complex FFT, followed by IFFT and checks that the
 * original array has been returned.
 */
void testFFTW1DComplex() 
{
    // Problem size
    int N = 128;
    
    // Allocate values in and out
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *backin = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    
    // Input values
    for (int i=0; i<N; i++) {
        in[i][0] = std::sin(2*i*M_PI/N);
        in[i][1] = std::cos(2*i*M_PI/N);
    }
    
    // FFT
    
    // Execution plan
    fftw_plan forward = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(forward);
    
    // Scale correctly
    for (int i=0; i<N; i++) {
        out[i][0] /= N;
        out[i][1] /= N;
    }
    
    // IFFT
    fftw_plan backward = fftw_plan_dft_1d(N, out, backin, FFTW_BACKWARD, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(backward);
    
    // Compare
    for (int i=0; i<N; i++) {
        if (std::abs(in[i][0]-backin[i][0]) > closeEnough) {
            std::cerr << "Re(in) = " << in[i][0] << std::endl;
            std::cerr << "Re(backin) = " << backin[i][0] << std::endl;
            throw std::runtime_error("Real parts didn't match.");
        }
        if (std::abs(in[i][1]-backin[i][1]) > closeEnough) {
            std::cerr << "Im(in) = " << in[i][1] << std::endl;
            std::cerr << "Im(backin) = " << backin[i][1] << std::endl;
            throw std::runtime_error("Imaginary parts didn't match.");
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
 * @brief Computes a 1D real FFT, followed by IFFT and checks that the
 * original array has been returned.
 */
void testFFTW1DReal() 
{
    // Problem size
    int N = 128;
    int NComplex = N/2+1;
    
    // Allocate values in and out
    double *in = (double*) fftw_malloc(sizeof(double) * N);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NComplex);
    double *backin = (double*) fftw_malloc(sizeof(double) * N);
    
    // Input values
    for (int i=0; i<N; i++) {
        in[i] = std::sin(2*i*M_PI/N);
    }
    
    // FFT
    
    // Execution plan
    fftw_plan forward = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(forward);
    
    // Scale correctly
    for (int i=0; i<NComplex; i++) {
        out[i][0] /= N;
        out[i][1] /= N;
    }
    
    // IFFT
    fftw_plan backward = fftw_plan_dft_c2r_1d(N, out, backin, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(backward);
    
    // Compare
    for (int i=0; i<N; i++) {
        if (std::abs(in[i]-backin[i]) > closeEnough) {
            std::cerr << "In testFFTW1DReal" << std::endl;
            std::cerr << "in = " << in[i] << std::endl;
            std::cerr << "backin = " << backin[i] << std::endl;
            throw std::runtime_error("Real part didn't match.");
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
 */
void testFFTW2DReal() 
{
    // Problem size
    int N1 = 128;
    int N2 = 128;
    int NReal = N1*N2;
    int NComplex = N1*(N2/2+1);
    
    // Allocate values in and out
    double *in = (double*) fftw_malloc(sizeof(double) * NReal);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NComplex);
    double *backin = (double*) fftw_malloc(sizeof(double) * NReal);
    
    // Input values
    for (int i=0; i<N1; i++) {
        for (int j=0; j<N2; j++) {
            in[hpp::fftwFlat2(i,j,N1,N2)] = std::sin(2*i*M_PI/N1) + std::cos(2*j*M_PI/N2);
        }
    }
    
    // FFT
    
    // Execution plan
    fftw_plan forward = fftw_plan_dft_r2c_2d(N1, N2, in, out, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(forward);
    
    // Scale correctly
    for (int i=0; i<NComplex; i++) {
        out[i][0] /= NReal;
        out[i][1] /= NReal;
    }
    
    // IFFT
    fftw_plan backward = fftw_plan_dft_c2r_2d(N1, N2, out, backin, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(backward);
    
    // Compare
    for (int idx=0; idx<NReal; idx++) {
        if (std::abs(in[idx]-backin[idx]) > closeEnough) {
            std::cerr << "In testFFTW2DReal" << std::endl;
            std::cerr << "in = " << in[idx] << std::endl;
            std::cerr << "backin = " << backin[idx] << std::endl;
            throw std::runtime_error("Real part didn't match.");
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
 * @brief Computes a 1D real FFT, followed by IFFT and checks that the
 * original array has been returned.
 */
void testFFTWNDReal(int rank) 
{
    // Problem size
    int N = 16;
    std::vector<int> realDims(rank, N);
    std::vector<int> complexDims(rank, N);
    complexDims.back() = N/2+1;
    int NReal = 1;
    for (auto dim : realDims) {
        NReal *= dim;
    }
    int NComplex = 1;
    for (auto dim : complexDims) {
        NComplex *= dim;
    }
    
    // Allocate values in and out
    double *in = (double*) fftw_malloc(sizeof(double) * NReal);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NComplex);
    double *backin = (double*) fftw_malloc(sizeof(double) * NReal);
    
    // Input values
    for (int idx=0; idx<NReal; idx++) {
        in[idx] = idx;
    }
    
    // FFT
    
    // Execution plan
    fftw_plan forward = fftw_plan_dft_r2c(rank, realDims.data(), in, out, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(forward);
    
    // Scale correctly
    for (int i=0; i<NComplex; i++) {
        out[i][0] /= NReal;
        out[i][1] /= NReal;
    }
    
    // IFFT
    fftw_plan backward = fftw_plan_dft_c2r(rank, realDims.data(), out, backin, FFTW_ESTIMATE);
    
    // Execute
    fftw_execute(backward);
    
    // Compare
    for (int idx=0; idx<NReal; idx++) {
        if (std::abs(in[idx]-backin[idx]) > closeEnough) {
            std::cerr << "In testFFTWNDReal" << std::endl;
            std::cerr << "in = " << in[idx] << std::endl;
            std::cerr << "backin = " << backin[idx] << std::endl;
            throw std::runtime_error("Real part didn't match.");
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
    // 1D
    testFFTW1DComplex();
    testFFTW1DReal();
    
    // 2D
    testFFTW2DReal();
    
    // 4D
    testFFTWNDReal(4);

    // Return
    return 0;
}