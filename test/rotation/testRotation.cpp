#include <stdexcept>
#include <hpp/rotation.h>

namespace hpp {

template<typename T> void getAxisAngle(Tensor2<T> R, std::vector<T>& u, T& theta){
    u[0] = R(2,1)-R(1,2);
    u[1] = R(0,2)-R(2,0);
    u[2] = R(1,0)-R(0,1);
    T r = std::sqrt(std::pow(u[0],(T)2.0)+std::pow(u[1],(T)2.0)+std::pow(u[2],(T)2.0));
    T t = R.tr();
    theta = std::atan2(r, t-1);
}

template<typename T>
void testConversions(){
   // Rotations
    EulerAngles<T> angle;
    T angleEquiv = 1e6*std::numeric_limits<T>::epsilon();
    int NPointsPerAngle = 16;
    for (int i=0; i<NPointsPerAngle; i++) {
        angle.alpha = i*(2*M_PI/NPointsPerAngle);
        for (int j=0; j<NPointsPerAngle; j++) {
            angle.beta = j*(M_PI/NPointsPerAngle)+0.001;
            for (int k=0; k<NPointsPerAngle; k++) {
                angle.gamma = k*(2*M_PI/NPointsPerAngle);
                Tensor2<T> R = EulerZXZRotationMatrix(angle);
                EulerAngles<T> angleRestored = getEulerZXZAngles(R);
                bool angleGood = true;
                if (std::abs(angle.alpha-angleRestored.alpha) > angleEquiv) angleGood = false;
                if (std::abs(angle.beta-angleRestored.beta) > angleEquiv) angleGood = false;
                if (std::abs(angle.gamma-angleRestored.gamma) > angleEquiv) angleGood = false;
                if (!angleGood) {
                    std::cerr << "Input: " << angle.alpha << " " << angle.beta << " " << angle.gamma << std::endl;
                    std::cerr << "Output: " << angleRestored.alpha << " " << angleRestored.beta << " " << angleRestored.gamma << std::endl;
                    throw std::runtime_error("Angles don't match.");
                }
            }
        }
    }  
}

template<typename T>
void testRandom() {
    // Do an axis-angle decomposition, and ensure that the axis is uniformly
    // distributed, and the angle is distributed with CDF (1/pi)(theta-sin(theta))
    // i.e. PDF (1/pi)(1-cos(theta))
    Tensor2<T> R(3,3);
    std::vector<T> u(3);
    T theta;
    
    // Set up histogram
    int nBins = 100;
    int nSamples = 1000000;
    T binWidthTheta = M_PI/(nBins);
    std::valarray<T> histTheta(nBins);
    histTheta = 0.0;
    
    // Samples
    for (int iS=0; iS<nSamples; iS++) {
        randomRotationTensorInPlace<T>(3, R, true);
        getAxisAngle(R, u, theta);
        int iTheta = (int)(theta/binWidthTheta);
        histTheta[iTheta] += 1.0;
    }
    histTheta /= (T)nSamples;
    
    // Analytic density
    std::valarray<T> histAnalyticTheta(nBins);
    for (int iBin=0; iBin<nBins; iBin++) {
        theta = (iBin+0.5)*binWidthTheta;
        histAnalyticTheta[iBin] = binWidthTheta*(1.0/M_PI)*(1.0-std::cos(theta));
    }
    
    // Theta error
    std::valarray<T> histErrorTheta = std::abs(histTheta-histAnalyticTheta)/histAnalyticTheta;
    T meanErrorTheta = 0.0;
    for (auto err : histErrorTheta) {
        meanErrorTheta += err;
    }
    meanErrorTheta /= histErrorTheta.size();
    if (meanErrorTheta > 0.03) {
        std::cerr << "Mean error: " << meanErrorTheta << std::endl;
        throw std::runtime_error("Mean error in random rotation is too large.");
    }
}

/**
 * @brief Checks known correct angles are produced by the discrete orientation spaces.
 */
template <typename U>
void testSpaces() 
{
    std::vector<int> indices = {0,12345,23456};
    std::vector<EulerAngles<U>> knownValuesResolution3(3);
    knownValuesResolution3[0] = EulerAngles<U>(0.712446,0.716902,1.46003);
    knownValuesResolution3[1] = EulerAngles<U>(1.58868,0.197944,2.10593);
    knownValuesResolution3[2] = EulerAngles<U>(-2.60062,-0.521592,1.66706);

    hpp::SO3Discrete<U> so3(3);
    for (unsigned int i=0; i<3; i++) {
        auto knownAngle = knownValuesResolution3[i];
        auto producedAngle = so3.getEulerAngle(indices[i]);
        U err = 0.0;
        err += std::pow(knownAngle.alpha - producedAngle.alpha, 2.0);
        err += std::pow(knownAngle.beta - producedAngle.beta, 2.0);
        err += std::pow(knownAngle.gamma - producedAngle.gamma, 2.0);
        err = std::sqrt(err);
        if (err > 1e-5) {
            std::cerr << "known angle = " << knownAngle << std::endl;
            std::cerr << "produced angle = " << producedAngle << std::endl;
            std::cerr << "error = " << err << std::endl;
            throw std::runtime_error("Produced angle doesn't match known angle.");
        }
    }
}

} //END NAMESPACE HPP

int main(int argc, char *argv[]) {
    hpp::testConversions<float>();
    hpp::testConversions<double>();
    hpp::testRandom<float>();
    hpp::testRandom<double>();
    hpp::testSpaces<float>();
    hpp::testSpaces<double>();
    return 0;
}