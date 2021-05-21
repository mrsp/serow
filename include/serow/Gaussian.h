/* 
 * Copyright 2017-2021 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH) 
 *		 nor the names of its contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GAUSSIAN_H
#define GAUSSIAN_H
#include <cmath>
class Gaussian {
  public:
    /** @fn double pdf(double x)
     *  @brief Computes the Standard Gaussian pdf 
     *  @param x Real variable
     *  @return  Standard Gaussian pdf value
     */
    double pdf(double x) {
        return exp(-x*x / 2.00) / sqrt(2 * 3.141592653589793238463);
    }
    /** @fn double pdf(double x, double mu, double sigma)
     *  @brief Computes the Gaussian pdf with mean mu and stddev sigma
     *  @param x Real variable
     *  @param mu mean value of the Gaussian distribution
     *  @param sigma std of the Gaussian distribution
     *  @return  Gaussian pdf value
     */
    double pdf(double x, double mu, double sigma) {
        return pdf((x - mu) / sigma) / sigma;
    }
    /** @fn double cdf(double z)
     *  @brief Computes the standard Gaussian cdf using Taylor approximation
     *  @param z Real variable
     *  @return Standard Gaussian cdf value
     */
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
    /** @fn double cdf(double z, double mu, double sigma)
     *  @brief Computes the  Gaussian cdf with mean mu and stddev sigma
     *  @param z Real variable
     *  @param mu mean value of the Gaussian distribution
     *  @param sigma std of the Gaussian distribution
     *  @return Gaussian cdf value
     */
    double cdf(double z, double mu, double sigma) {
        return cdf((z - mu) / sigma);
    } 
};
#endif