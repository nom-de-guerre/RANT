/*

Copyright (c) 2020, Douglas Santry
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, is permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef __ANT__COMMON__H__
#define __ANT__COMMON__H__

#include <assert.h>

#ifndef IEEE_t
typedef double IEEE_t;
#endif

#define RECTIFIER(X) log (1 + exp (X))
#define SIGMOID_FN(X) (1 / (1 + exp (-X))) // derivative of rectifier
#define SIGMOID_DERIV(Y) (Y * (1 - Y))
#define RELU(X) ((X) < 0.0 ? 0.0 : (X))
#define RELU_DERIVATIVE_FN(Y) (Y > 0.0 ? 1.0 : 0.0)

#ifdef __TANH_ACT_FN
#define ACTIVATION_FN(X) tanh(X)
#define DERIVATIVE_FN(Y) (1 - Y*Y)
#elif defined (__RELU)
#define ACTIVATION_FN(X) RELU(X)
#define DERIVATIVE_FN(Y) RELU_DERIVATIVE_FN(Y)
#elif defined(__IDENTITY)
#define ACTIVATION_FN(X) (X)
#define DERIVATIVE_FN(Y) 1.0
#else
#define ACTIVATION_FN(X) SIGMOID_FN(X)
#define DERIVATIVE_FN(Y) SIGMOID_DERIV(Y)
#endif

#endif 

