//
// MATLAB Compiler: 7.0.1 (R2019a)
// Date: Tue Oct 21 15:20:33 2025
// Arguments:
// "-B""macro_default""-W""cpplib:pcgsolver""-T""link:lib""pcgsolver.m""-C"
//

#ifndef __pcgsolver_h
#define __pcgsolver_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#include "mclcppclass.h"
#ifdef __cplusplus
extern "C" {
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_pcgsolver_C_API 
#define LIB_pcgsolver_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_pcgsolver_C_API 
bool MW_CALL_CONV pcgsolverInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_pcgsolver_C_API 
bool MW_CALL_CONV pcgsolverInitialize(void);

extern LIB_pcgsolver_C_API 
void MW_CALL_CONV pcgsolverTerminate(void);

extern LIB_pcgsolver_C_API 
void MW_CALL_CONV pcgsolverPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_pcgsolver_C_API 
bool MW_CALL_CONV mlxPcgsolver(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#ifdef __cplusplus
}
#endif


/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__MINGW64__)

#ifdef EXPORTING_pcgsolver
#define PUBLIC_pcgsolver_CPP_API __declspec(dllexport)
#else
#define PUBLIC_pcgsolver_CPP_API __declspec(dllimport)
#endif

#define LIB_pcgsolver_CPP_API PUBLIC_pcgsolver_CPP_API

#else

#if !defined(LIB_pcgsolver_CPP_API)
#if defined(LIB_pcgsolver_C_API)
#define LIB_pcgsolver_CPP_API LIB_pcgsolver_C_API
#else
#define LIB_pcgsolver_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_pcgsolver_CPP_API void MW_CALL_CONV pcgsolver(int nargout, mwArray& result, const mwArray& A, const mwArray& B);

/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */
#endif

#endif
