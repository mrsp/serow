/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
/**
 * @brief Numerical Differentiation with the First Order Euler Method
 * @author Stylianos Piperakis
 * @details to numerical compute the derivative of a signal
 */

#pragma once
#include <string>

namespace serow {

class Differentiator {
   public:
    /** @fn void setParams(double dt)
     *  @brief differentiates the measurement with finite differences
     *  @param dt Sampling time in seconds e.g. 0.01s
     */
    void setParams(double dt) { dt_ = dt; }
    /** @fn void filter(double x)
     *  @brief differentiates the measurement with finite differences
     *  @param x signal to be differentiatied
     */
    double filter(double x);

    /** @fn void init(std::string name, double dt);
     *  @brief initializes the numerical differentiator
     *  @param name name of the signal e.g LHipYawPitch
     *  @param dt Sampling time in seconds e.g. 0.01s
     */
    void init(std::string name, double dt);

    /** @fn void reset();
     *  @brief  resets the the numerical differentiator's state
     */
    void reset();

   private:
    double x_prev_{};
    double xdot_{};
    double dt_{};
    bool firstrun_{true};
    std::string name_{};
};

} // namespace serow
