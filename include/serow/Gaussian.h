#ifndef GAUSSIAN_H
#define GAUSSIAN_H
#include <cmath>
class Gaussian {
  public:
    // return pdf(x) = standard Gaussian pdf
    double pdf(double x) {
        return exp(-x*x / 2.00) / sqrt(2 * 3.141592653589793238463);
    }

    // return pdf(x, mu, signma) = Gaussian pdf with mean mu and stddev sigma
    double pdf(double x, double mu, double sigma) {
        return pdf((x - mu) / sigma) / sigma;
    }

    // return cdf(z) = standard Gaussian cdf using Taylor approximation
    double cdf(double z) {
        if (z < -8.0) return 0.0;
        if (z >  8.0) return 1.0;
        double sum = 0.0, term = z;
        for (int i = 3; sum + term != sum; i += 2) {
            sum  = sum + term;
            term = term * z * z / i;
        }
        return 0.5 + sum * pdf(z);
    }

    // return cdf(z, mu, sigma) = Gaussian cdf with mean mu and stddev sigma
    double cdf(double z, double mu, double sigma) {
        return cdf((z - mu) / sigma);
    } 
};
#endif