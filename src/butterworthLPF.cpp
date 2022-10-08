/*
 * Copyright 2017-2023 Stylianos Piperakis,
 * Foundation for Research and Technology Hellas (FORTH)
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
 *     * Neither the name of the Foundation for Research and Technology Hellas
 *       (FORTH) nor the names of its contributors may be used to endorse or
 *       promote products derived from this software without specific prior
 *       written permission.
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

#include <serow/butterworthLPF.h>

butterworthLPF::butterworthLPF()
{
    fx = 0;
    fs = 0;
    a1 = 0;
    a2 = 0;
    b0 = 0;
    b1 = 0;
    b2 = 0;
    ff = 0;
    ita = 0;
    q = 0;
    i = 0;
    y_p = 0;
    y_pp = 0;
    x_p = 0;
    x_pp = 0;
}

void butterworthLPF::reset()
{
    fx = 0;
    fs = 0;
    a1 = 0;
    a2 = 0;
    b0 = 0;
    b1 = 0;
    b2 = 0;
    ff = 0;
    ita = 0;
    q = 0;
    i = 0;
    y_p = 0;
    y_pp = 0;
    x_p = 0;
    x_pp = 0;
    std::cout<<name<<"Low-pass Butterworth filter reset"<<std::endl;
}

void butterworthLPF::init(std::string name_, double fsampling, double fcutoff)
{
    fs = fsampling;
    fx = fcutoff;

    ff = fx / fs;
    ita = 1.0 / tan(3.14159265359 * ff);
    q = sqrt(2.0);
    b0 = 1.0 / (1.0 + q * ita + ita * ita);
    b1 = 2 * b0;
    b2 = b0;
    a1 = 2.0 * (ita * ita - 1.0) * b0;
    a2 = -(1.0 - q * ita + ita * ita) * b0;
    name = name_;
    a = (2.0 * 3.14159265359 * ff) / (2.0 * 3.14159265359 * ff + 1.0);
    std::cout<<name<<"Low-pass Butterworth filter initialized"<<std::endl;
}

double butterworthLPF::filter(double y)
{
    double out;
    if (i > 2)
    {
        out = b0 * y + b1 * y_p + b2 * y_pp + a1 * x_p + a2 * x_pp;
    }
    else
    {
        out = x_p + a * (y - x_p);
        i++;
    }

    y_pp = y_p;
    y_p = y;
    x_pp = x_p;
    x_p = out;

    return out;
}
