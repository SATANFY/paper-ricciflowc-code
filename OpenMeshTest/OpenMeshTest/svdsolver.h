//
// MATLAB Compiler: 7.0.1 (R2019a)
// Date: Thu Oct 16 22:03:23 2025
// Arguments:
// "-B""macro_default""-W""cpplib:svdsolver""-T""link:lib""svdsolver.m""-C"
//

#ifndef __svdsolver_h
#define __svdsolver_h 1

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
#ifndef LIB_svdsolver_C_API 
#define LIB_svdsolver_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_svdsolver_C_API 
bool MW_CALL_CONV svdsolverInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_svdsolver_C_API 
bool MW_CALL_CONV svdsolverInitialize(void);

extern LIB_svdsolver_C_API 
void MW_CALL_CONV svdsolverTerminate(void);

extern LIB_svdsolver_C_API 
void MW_CALL_CONV svdsolverPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_svdsolver_C_API 
bool MW_CALL_CONV mlxSvdsolver(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#ifdef __cplusplus
}
#endif


/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__MINGW64__)

#ifdef EXPORTING_svdsolver
#define PUBLIC_svdsolver_CPP_API __declspec(dllexport)
#else
#define PUBLIC_svdsolver_CPP_API __declspec(dllimport)
#endif

#define LIB_svdsolver_CPP_API PUBLIC_svdsolver_CPP_API

#else

#if !defined(LIB_svdsolver_CPP_API)
#if defined(LIB_svdsolver_C_API)
#define LIB_svdsolver_CPP_API LIB_svdsolver_C_API
#else
#define LIB_svdsolver_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_svdsolver_CPP_API void MW_CALL_CONV svdsolver(int nargout, mwArray& result, const mwArray& A, const mwArray& B);

/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */
#endif

#endif
