//
// include necessary system headers
//
#include <cmath>
#include <mex.h>
#include <array>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <iostream>


// INPUTS:
//  - cell_diagonal<double> [nc x m] if column major or [m x nc] if row major)
//  - N<double>             [nf x 2]
//  - nc<double>            [scalar]
// OUTPUT:
//  - face_diagonal<double> [nf x 2*m] if column major or [2*m x nf] if row major
const char* inputCheck(const int nin, const int nout, int & status_code){
    if (nin == 0) {
        if (nout > 0) {
            status_code = -1;
            return "Cannot give outputs with no inputs.";
        }
        // We are being called through compilation testing. Just do nothing.
        // If the binary was actually called, we are good to go.
        status_code = 1;
        return "";
    } else if (nin != 4) {
        status_code = -2;
        return "4 input arguments required: Diagonal, N, number of cells and rowMajor indicator";
    } else if (nout > 1) {
        status_code = -3;
        return "Too many outputs requested. Function has a single output argument.";
    } else {
        // All ok.
        status_code = 0;
        return "";
    }
}

const char* dimensionCheck(const int nc, const int nrows, const int ncols, int & status_code){
    if(nrows != nc && ncols != nc){
        status_code = -5;
        return "Malformed input. No dimension of input matrix matches number of cells: Dimensions of diagonal matrix does not fit either RowMajor or ColMajor";
    }
}

template <bool rowMajor>
void faceAverageJac(const int nf, const int nc, const int m, const double* diagonal, const double* N, double* result) {
#pragma omp parallel
    for (int face = 0; face < nf; face++) {
        int left = N[face] - 1;
        int right = N[face + nf] - 1;
        for (int der = 0; der < m; der++) {
            if (rowMajor) {
                result[2 * m * face + der] = 0.5 * diagonal[m * left + der];
                result[2 * m * face + der + m] = 0.5 * diagonal[m * right + der];
            }
            else {
                result[der * nf + face] = 0.5 * diagonal[nc * der + left];
                result[der * nf + face + m * nf] = 0.5 * diagonal[nc * der + right];
            }
        }
    }
}


template <int m, bool rowMajor>
void faceAverageJac(const int nf, const int nc, const double* diagonal, const double* N, double* result) {
    faceAverageJac<rowMajor>(nf, nc, m, diagonal, N, result);
}

template <bool rowMajor>
void faceAverageJacMain(const int m, const int nf, const int nc, const double * diagonal, const double * N, double * result){
        switch (m) {
            case 1:
                faceAverageJac<1, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 2:
                faceAverageJac<2, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 3:
                faceAverageJac<3, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 4:
                faceAverageJac<4, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 5:
                faceAverageJac<5, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 6:
                faceAverageJac<6, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 7:
                faceAverageJac<7, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 8:
                faceAverageJac<8, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 9:
                faceAverageJac<9, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 10:
                faceAverageJac<10, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 11:
                faceAverageJac<11, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 12:
                faceAverageJac<12, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 13:
                faceAverageJac<13, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 14:
                faceAverageJac<14, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 15:
                faceAverageJac<15, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 16:
                faceAverageJac<16, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 17:
                faceAverageJac<17, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 18:
                faceAverageJac<18, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 19:
                faceAverageJac<19, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 20:
                faceAverageJac<20, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 21:
                faceAverageJac<21, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 22:
                faceAverageJac<22, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 23:
                faceAverageJac<23, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 24:
                faceAverageJac<24, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 25:
                faceAverageJac<25, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 26:
                faceAverageJac<26, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 27:
                faceAverageJac<27, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 28:
                faceAverageJac<28, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 29:
                faceAverageJac<29, rowMajor>(nf, nc, diagonal, N, result);
                break;
            case 30:
                faceAverageJac<30, rowMajor>(nf, nc, diagonal, N, result);
                break;
            default:
                faceAverageJac<rowMajor>(nf, nc, m, diagonal, N, result);
        }
}


/* MEX gateway */
void mexFunction( int nlhs,       mxArray *plhs[], 
		          int nrhs, const mxArray *prhs[] )
     
{ 
    int status_code = 0;
    auto msg = inputCheck(nrhs, nlhs, status_code);
    if(status_code < 0){
        // Some kind of error
        mexErrMsgTxt(msg);
    } else if (status_code == 1){
        // Early return
        return;
    }
    double * diagonal = mxGetPr(prhs[0]);
    double * N = mxGetPr(prhs[1]);
    bool rowMajor = mxGetScalar(prhs[3]);

    int nc = mxGetScalar(prhs[2]);
    int nf = mxGetM(prhs[1]);

    // Dimensions of diagonals - figure out if we want row or column major solver
    int nrows = mxGetM(prhs[0]);
    int ncols = mxGetN(prhs[0]);

    int status_code2 = 0;
    auto msg2 = dimensionCheck(nc, nrows, ncols, status_code2);
    if(status_code2 < 0){
        // Some kind of error
        mexErrMsgTxt(msg2);
    }
    if (nrows == nc) {
        // ColMajor
        int m = ncols;
        plhs[0] = mxCreateDoubleMatrix(nf, 2 * m, mxREAL);
        double* result = mxGetPr(plhs[0]);
        faceAverageJacMain<false>(m, nf, nc, diagonal, N, result);
    }
    else if (ncols == nc){
        // RowMajor
        int m = nrows;
        plhs[0] = mxCreateDoubleMatrix(2 * m, nf, mxREAL);
        double* result = mxGetPr(plhs[0]);
        faceAverageJacMain<true>(m, nf, nc, diagonal, N, result);
    }
    return;
}


