#include "pcgsolver.h"
#include "svdsolver.h"
#include "matricsolver.h"
#include "lsqminnormsolver.h"
#include <cmath>
#include <vector>
#include <queue>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers> 
#include <opencv2/opencv.hpp>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

using namespace std;
using namespace cv;
using namespace Eigen;

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

// 边界自由 & 终止条件 & 最大循环
#define BOUNDARYFREE false
#define EPSILON 0.0001
#define TIMES 50
// 运算引擎 opencv & eigen & matlab
#define ENGINE "matlab"
// 矩阵方法 lsqminnorm & svd & pcg
#define METHOD "pcg"

/*------------校验函数------------*/

// 数据检查
void icheck(int* Vector, int i_max, int row = 12) {
	std::cout << "****************************************" << endl;
	for (size_t i = 0; i < i_max; i++) {
		std::cout << Vector[i] << " ";
		if ((i + 1) % row == 0) std::cout << endl;
	}
	std::cout << endl;
}
void icheck(double* Vector, int i_max, int row = 12) {
	std::cout << "****************************************" << endl;
	for (size_t i = 0; i < i_max; i++){
		std::cout << Vector[i] << " ";
		if ((i + 1) % row == 0) std::cout << endl;
	}
	std::cout << endl;
}
void icheck(Mat Matrix, int i_max, int j_max, int row = 12) {
	std::cout << "****************************************" << endl << "****************************************" << endl;
	for (size_t i = 0; i < i_max; i++) {
		for (size_t j = 0; j < j_max; j++) {
			// 向量输出
			if ((Matrix.at<double>(i, j) != 0) && (j_max == 1)) std::cout << Matrix.at<double>(i, j) << " ";
			// 异常数据标识定位
			if ((abs(Matrix.at<double>(i, j)) > 1) && (j_max == 1)) std::cout << "	==========>>	" << i;
			// 矩阵输出
			if ((Matrix.at<double>(i, j) != 0) && (j_max > 1)) std::cout << Matrix.at<double>(i, j) << " ";
			if (((j + 1) % row == 0) && (j_max > 1)) std::cout << endl;
		}
		std::cout << endl << endl;
	}
	std::cout << endl;
}
void icheck(vector<vector<double>> Matrix, int i_max, int j_max, int row = 12) {
	std::cout << "****************************************" << endl << "****************************************" << endl;
	for (size_t i = 0; i < i_max; i++){
		for (size_t j = 0; j < j_max; j++) {
			std::cout << Matrix[i][j] << " ";
			if ((j + 1) % row == 0) std::cout << endl;
		}
		std::cout << endl << endl;
	}
	std::cout << endl;
}

/*-----------初始化函数-----------*/

// 半径计算
void get_gama(MyMesh mesh, double*& Gama, Mat& U) {
	for (const auto& fh : mesh.faces()) {
		MyMesh::HalfedgeHandle h = fh.halfedge();
		int v1 = mesh.from_vertex_handle(h).idx();
		int v2 = mesh.from_vertex_handle(mesh.next_halfedge_handle(h)).idx();
		int v3 = mesh.from_vertex_handle(mesh.prev_halfedge_handle(h)).idx();
		double l1_2 = mesh.calc_edge_length(h);
		double l2_3 = mesh.calc_edge_length(mesh.next_halfedge_handle(h));
		double l3_1 = mesh.calc_edge_length(mesh.prev_halfedge_handle(h));
		// 确定顶点圆 ci 半径
		if (Gama[v1] == 0 || Gama[v1] > (l1_2 + l3_1 - l2_3) / 2.0) Gama[v1] = (l1_2 + l3_1 - l2_3) / 2.0;
		if (Gama[v2] == 0 || Gama[v2] > (l1_2 + l2_3 - l3_1) / 2.0) Gama[v2] = (l1_2 + l2_3 - l3_1) / 2.0;
		if (Gama[v3] == 0 || Gama[v3] > (l2_3 + l3_1 - l1_2) / 2.0) Gama[v3] = (l2_3 + l3_1 - l1_2) / 2.0;
	}
	for (size_t i = 0; i < mesh.n_vertices(); i++) U.at<double>(i, 0) = log(Gama[i]);
	return;
}

// 距离计算（返回边界边总长 & 非欧情形提示）
double get_distance(MyMesh mesh, double* Gama, double*& I, double*& L, int*& F) {
	double boundarylength = 0;
	for (const auto& eh : mesh.edges()) {
		double r1 = Gama[eh.v0().idx()], r2 = Gama[eh.v1().idx()], l = mesh.calc_edge_length(eh);
		double cosl = (r1 * r1 + r2 * r2 - l * l) / (2.0 * r1 * r2);
		if (mesh.is_boundary(eh)) boundarylength += mesh.calc_edge_length(eh);
		if (cosl < 0) {
			I[eh.idx()] = -cosl;
			L[eh.idx()] = sqrt(r1 * r1 + r2 * r2 + 2.0 * I[eh.idx()] * r1 * r2);
		} 
		else {
			I[eh.idx()] = (cosh(l) - cosh(r1) * cosh(r2)) / (sinh(r1) * sinh(r2));
			L[eh.idx()] = 1.0 / cosh(cosh(r1) * cosh(r2) + I[eh.idx()] * sinh(r1) * sinh(r2));
			std::cout << endl << "| ******************************** WARNING ******************************** |" << endl;
			std::cout << endl << "| ** The Intersection Angles are OBTUSE, The Solution May Not be UNIQUE! ** |" << endl;
			std::cout << endl << "| ******* The Conformality of The Final Mapping May be COMPROMISED! ******* |" << endl;
			std::cout << endl << "| ************************************************************************* |" << endl;
			for (const auto& f : eh.faces()) F[f.idx()]++;
		}
	}
	return boundarylength;
}

// 夹角计算
void get_theta(MyMesh mesh, double* L, int* F, vector<vector<double>>& Theta) {
	for (const auto& fh : mesh.faces()) {
		int i = 0, l1_2 = 0, l2_3 = 0, l3_1 = 0;
		/*
		MyMesh::HalfedgeHandle h = fh.halfedge();
		std::cout << "Halfedge: " << h.idx() << "-h vf : " << mesh.from_vertex_handle(h) << " vt : " << mesh.to_vertex_handle(h) << endl;
		MyMesh::HalfedgeHandle nh = mesh.next_halfedge_handle(h);
		std::cout << "Halfedge: " << nh.idx() << "-nh vf : " << mesh.from_vertex_handle(nh) << " vt : " << mesh.to_vertex_handle(nh) << endl;
		MyMesh::HalfedgeHandle nnh = mesh.next_halfedge_handle(nh);
		std::cout << "Halfedge: " << nnh.idx() << "-nnh vf : " << mesh.from_vertex_handle(nnh) << " vt : " << mesh.to_vertex_handle(nnh) << endl;
		MyMesh::HalfedgeHandle ph = mesh.prev_halfedge_handle(h);
		std::cout << "Halfedge: " << ph.idx() << "-ph vf : " << mesh.from_vertex_handle(ph) << " vt : " << mesh.to_vertex_handle(ph) << endl;
		*/
		for (const auto& e : fh.edges()) {
			//std::cout <<"Face: "<< fh.idx() << "-" << i << " v0 :" << e.v0().idx() << " v1 :" << e.v1().idx() << endl;
			if (i == 0) l1_2 = e.idx();
			else if (i == 1) l2_3 = e.idx();
			else l3_1 = e.idx();
			i++;
		}
		if (F[fh.idx()] == 0) {
			Theta[fh.idx()][0] = acos((L[l1_2] * L[l1_2] + L[l3_1] * L[l3_1] - L[l2_3] * L[l2_3]) / (2.0 * L[l1_2] * L[l3_1]));
			Theta[fh.idx()][1] = acos((L[l1_2] * L[l1_2] + L[l2_3] * L[l2_3] - L[l3_1] * L[l3_1]) / (2.0 * L[l1_2] * L[l2_3]));
			Theta[fh.idx()][2] = acos((L[l3_1] * L[l3_1] + L[l2_3] * L[l2_3] - L[l1_2] * L[l1_2]) / (2.0 * L[l3_1] * L[l2_3]));
		}
		else {
			Theta[fh.idx()][0] = acos((cosh(L[l2_3]) - cosh(L[l1_2]) * cosh(L[l3_1])) / (sinh(L[l1_2]) * sinh(L[l3_1])));
			Theta[fh.idx()][1] = acos((cosh(L[l3_1]) - cosh(L[l1_2]) * cosh(L[l2_3])) / (sinh(L[l1_2]) * sinh(L[l2_3])));
			Theta[fh.idx()][2] = acos((cosh(L[l1_2]) - cosh(L[l3_1]) * cosh(L[l2_3])) / (sinh(L[l3_1]) * sinh(L[l2_3])));
		}
	}
	return;
}

// 偏导计算
void get_round(MyMesh mesh, double* Gama, vector<vector<double>>& Del) {
	int v1, v2, v3;
	double l1_2, l2_3, l3_1, theta, x;
	for (const auto& fh : mesh.faces()) {
		MyMesh::HalfedgeHandle h = fh.halfedge();
		v1 = mesh.from_vertex_handle(h).idx();
		v2 = mesh.to_vertex_handle(h).idx();
		v3 = mesh.to_vertex_handle(mesh.next_halfedge_handle(h)).idx();
		l1_2 = mesh.calc_edge_length(h);
		l2_3 = mesh.calc_edge_length(mesh.next_halfedge_handle(h));
		l3_1 = mesh.calc_edge_length(mesh.prev_halfedge_handle(h));
		theta = acos((l1_2 * l1_2 + l3_1 * l3_1 - l2_3 * l2_3) / (2.0 * l1_2 * l3_1));
		x = (l1_2 * l1_2 + Gama[v1] * Gama[v1] - Gama[v2] * Gama[v2]) / (2.0 * l1_2);
		Del[fh.idx()][2] = (l3_1 * l3_1 - 2.0 * cos(theta) * l3_1 * x + Gama[v1] * Gama[v1] - Gama[v3] * Gama[v3]) / (2.0 * sin(theta) * l3_1) / l1_2;
		Del[fh.idx()][1] = sqrt(Del[fh.idx()][2] * l1_2 * Del[fh.idx()][2] * l1_2 + x * x) * sin(theta - acos(x / sqrt(Del[fh.idx()][2] * l1_2 * Del[fh.idx()][2] * l1_2 + x * x))) / l3_1;
		Del[fh.idx()][0] = 2.0 * (mesh.calc_face_area(fh) - 0.5 * Del[fh.idx()][2] * l1_2 * l1_2 - 0.5 * Del[fh.idx()][1] * l3_1 * l3_1) / l2_3 / l2_3;
	}
	return;
}

// 边界自由
void boundary_free(MyMesh mesh, Mat& K) {
	for (const auto& vh : mesh.vertices())
		if (mesh.is_boundary(vh))
			K.at<double>(vh.idx(), 0) = 0;
	return;
}

// 曲率计算
void get_gauss(MyMesh mesh, vector<vector<double>> Theta, Mat& K, double*& K_, double boundarylength, bool boundaryfree = BOUNDARYFREE) {
	for (const auto& vh : mesh.vertices()) {
		double k = 2.0 * M_PI;
		if (mesh.is_boundary(vh)) {
			k = M_PI;
			double l = 0;
			for (const auto& e : vh.edges()) {
				if (mesh.is_boundary(e)) l += mesh.calc_edge_length(e);
			}
			K_[vh.idx()] = l / boundarylength * M_PI * (mesh.n_vertices() + mesh.n_faces() - mesh.n_edges());
			//std::cout << "X[M] = " << mesh.n_vertices() + mesh.n_faces() - mesh.n_edges() << endl;
		}
		for (const auto& f : vh.faces()) {
			int i = 2;
			MyMesh::HalfedgeHandle h = f.halfedge();
			if (vh.idx() == mesh.from_vertex_handle(h).idx()) i = 0;
			else if (vh.idx() == mesh.to_vertex_handle(h).idx()) i = 1;
			k -= Theta[f.idx()][i];
		}
		K.at<double>(vh.idx(), 0) = K_[vh.idx()] - k;
	}
	if (boundaryfree) boundary_free(mesh, K);
	return;
}

/*-----------循环体函数-----------*/

// 矩阵计算
void get_delta(MyMesh mesh, vector<vector<double>> Del, Mat& Delta) {
	Delta = Mat_<double>::zeros(mesh.n_vertices(), mesh.n_vertices());
	int v1, v2, v3;
	for (const auto& fh : mesh.faces()) {
		MyMesh::HalfedgeHandle h = fh.halfedge();
		v1 = mesh.from_vertex_handle(h).idx();
		v2 = mesh.to_vertex_handle(h).idx();
		v3 = mesh.to_vertex_handle(mesh.next_halfedge_handle(h)).idx();
		Delta.at<double>(v1, v1) += Del[fh.idx()][2]; Delta.at<double>(v2, v2) += Del[fh.idx()][2]; Delta.at<double>(v1, v2) -= Del[fh.idx()][2]; Delta.at<double>(v2, v1) = Delta.at<double>(v1, v2);
		Delta.at<double>(v2, v2) += Del[fh.idx()][0]; Delta.at<double>(v3, v3) += Del[fh.idx()][0]; Delta.at<double>(v2, v3) -= Del[fh.idx()][0]; Delta.at<double>(v3, v2) = Delta.at<double>(v2, v3);
		Delta.at<double>(v3, v3) += Del[fh.idx()][1]; Delta.at<double>(v1, v1) += Del[fh.idx()][1]; Delta.at<double>(v3, v1) -= Del[fh.idx()][1]; Delta.at<double>(v1, v3) = Delta.at<double>(v3, v1);
	}
	return;
}

// 矩阵求解
Mat solve_mu(Mat A, Mat B, int n_v, string engine = ENGINE) {
	Mat Mu = Mat_<double>::zeros(n_v, 1);

	// Eigen
	if (engine == "eigen") {
		VectorXd KE(n_v); 
		VectorXd MuE(n_v);
		MatrixXd DeltaE(n_v, n_v);
		for (size_t i = 0; i < n_v; i++) KE(i) = B.at<double>(i, 0);
		for (size_t i = 0; i < n_v; i++) for (size_t j = 0; j < n_v; j++) DeltaE(i, j) = A.at<double>(i, j);
		// 完全主元变换秩 QR 分解 (适用)
		//MuE = DeltaE.fullPivHouseholderQr().solve(KE);
		// LDLT分解法 (不适用)
		LDLT<MatrixXd> ldlt;
		// 共轭梯度法 (不适用)
		ConjugateGradient<MatrixXd, Lower | Upper> cg;
		// 最小二乘共轭梯度法
		LeastSquaresConjugateGradient<MatrixXd> lscg;
		std::cout << "Start Solve Matrix by Eigen ..." << endl;
		lscg.compute(DeltaE);
		if (lscg.info() != Eigen::Success) cout << "Failed to Find the Result." << endl;
		else std::cout << "Find the Result." << endl;
		MuE = lscg.solve(KE);
		for (size_t i = 0; i < n_v; i++) Mu.at<double>(i, 0) = MuE(i);
	}
	// Matlab
	else if (engine == "matlab")
	{
		mwArray DeltaM(n_v, n_v, mxDOUBLE_CLASS);
		mwArray KM(n_v, 1, mxDOUBLE_CLASS);
		mwArray MuM(n_v, 1, mxDOUBLE_CLASS);
		for (size_t i = 0; i < n_v; i++) for (size_t j = 0; j < n_v; j++) DeltaM(i + 1, j + 1) = A.at<double>(i, j);
		for (size_t i = 0; i < n_v; i++) KM(i + 1, 1) = B.at<double>(i, 0);
		std::cout << "Start Solve Matrix by Matlab ..." << endl;
		// 最小范数最小二乘解
		if (METHOD == "lsqminnorm") lsqminnormsolver(1, MuM, DeltaM, KM);
		// SVD分解求解
		else if (METHOD == "svd") svdsolver(1, MuM, DeltaM, KM);
		// 预处理共轭梯度法
		else if (METHOD == "pcg") pcgsolver(1, MuM, DeltaM, KM);
		// 直接求解 (不适用)
		else matricsolver(1, MuM, DeltaM, KM);
		
		for (size_t i = 0; i < n_v; i++) Mu.at<double>(i, 0) = MuM(i + 1, 1);
	}
	// OpenCV
	else {
		// DECOMP_EIG & DECOMP_NORMAL & DECOMP_CHOLESKY
		if (!solve(A, B, Mu, DECOMP_SVD)) std::cout << "Failed to Find the Result." << endl;
		else std::cout << "Find the Result." << endl;
	}
	
	//icheck(Mu, n_v, 1);
	return Mu;
}

/*-----下列函数仅考虑欧氏情形-----*/

// 半径更新
void change_gama(double*& Gama, Mat U, int n_v) {
	for (size_t i = 0; i < n_v; i++) Gama[i] = exp(U.at<double>(i, 0));
	return;
}

// 距离更新
void change_distance(MyMesh mesh, double* Gama, double* I, double*& L) {
	for (const auto& eh : mesh.edges()) L[eh.idx()] = sqrt(Gama[eh.v0().idx()] * Gama[eh.v0().idx()] + Gama[eh.v1().idx()] * Gama[eh.v1().idx()] + 2.0 * I[eh.idx()] * Gama[eh.v0().idx()] * Gama[eh.v1().idx()]);
	return;
}

// 夹角更新
void change_theta(MyMesh mesh, double* L, vector<vector<double>>& Theta) {
	for (const auto& fh : mesh.faces()) {
		int i = 0, l1_2 = 0, l2_3 = 0, l3_1 = 0;
		for (const auto& e : fh.edges()) {
			if (i == 0) l1_2 = e.idx();
			else if (i == 1) l2_3 = e.idx();
			else l3_1 = e.idx();
			i++;
		}
		Theta[fh.idx()][0] = acos((L[l1_2] * L[l1_2] + L[l3_1] * L[l3_1] - L[l2_3] * L[l2_3]) / (2.0 * L[l1_2] * L[l3_1]));
		Theta[fh.idx()][1] = acos((L[l1_2] * L[l1_2] + L[l2_3] * L[l2_3] - L[l3_1] * L[l3_1]) / (2.0 * L[l1_2] * L[l2_3]));
		Theta[fh.idx()][2] = acos((L[l3_1] * L[l3_1] + L[l2_3] * L[l2_3] - L[l1_2] * L[l1_2]) / (2.0 * L[l3_1] * L[l2_3]));
	}
	return;
}

// 偏导更新
void change_round(MyMesh mesh, double* Gama, double* L, vector<vector<double>>& Del) {
	int v1 = 0, v2 = 0, v3 = 0;
	double l1_2 = 0, l2_3 = 0, l3_1 = 0;
	for (const auto& fh : mesh.faces()) {
		int i = 0;
		for (const auto& e : fh.edges()) {
			if (i == 0) {
				v1 = e.v0().idx();
				l1_2 = L[e.idx()];
			}
			else if (i == 1) {
				v2 = e.v0().idx();
				l2_3 = L[e.idx()];
			}
			else {
				v3 = e.v0().idx();
				l3_1 = L[e.idx()];
			}
			i++;
		}
		double theta = acos((l1_2 * l1_2 + l3_1 * l3_1 - l2_3 * l2_3) / (2.0 * l1_2 * l3_1));
		double iota = acos((l1_2 * l1_2 + l2_3 * l2_3 - l3_1 * l3_1) / (2.0 * l1_2 * l2_3));
		double x = (l1_2 * l1_2 + Gama[v1] * Gama[v1] - Gama[v2] * Gama[v2]) / (2.0 * l1_2);
		Del[fh.idx()][2] = (l3_1 * l3_1 - 2.0 * cos(theta) * l3_1 * x + Gama[v1] * Gama[v1] - Gama[v3] * Gama[v3]) / (2.0 * sin(theta) * l3_1) / l1_2;
		Del[fh.idx()][1] = sqrt(Del[fh.idx()][2] * l1_2 * Del[fh.idx()][2] * l1_2 + x * x) * sin(theta - acos(x / sqrt(Del[fh.idx()][2] * l1_2 * Del[fh.idx()][2] * l1_2 + x * x))) / l3_1;
		Del[fh.idx()][0] = sqrt(Del[fh.idx()][2] * l1_2 * Del[fh.idx()][2] * l1_2 + (l1_2 - x) * (l1_2 - x)) * sin(iota - acos((l1_2 - x) / sqrt(Del[fh.idx()][2] * l1_2 * Del[fh.idx()][2] * l1_2 + (l1_2 - x) * (l1_2 - x)))) / l2_3;
	}
	return;
}

// 曲率更新 ( return max| Ki_- Ki | )
double change_gauss(MyMesh mesh, vector<vector<double>> Theta, Mat& K, double* K_, bool boundaryfree = BOUNDARYFREE) {
	double e = 0;
	for (const auto& vh : mesh.vertices()) {
		double k = 2.0 * M_PI;
		if (mesh.is_boundary(vh)) k = M_PI;
		for (const auto& f : vh.faces()) {
			int i = 2;
			MyMesh::HalfedgeHandle h = f.halfedge();
			if (vh.idx() == mesh.from_vertex_handle(h).idx()) i = 0;
			else if (vh.idx() == mesh.to_vertex_handle(h).idx()) i = 1;
			k -= Theta[f.idx()][i];
		}
		K.at<double>(vh.idx(), 0) = K_[vh.idx()] - k;
		if (abs(K.at<double>(vh.idx(), 0)) > e) e = abs(K.at<double>(vh.idx(), 0));
	}
	if (boundaryfree) boundary_free(mesh, K);
	return e;
}

/*------------表面展平------------*/

class MyMeshFlattener {
public:
	// 更新队列
	static void triangle_queue(MyMesh mesh, OpenMesh::SmartFaceHandle fh, int& rear, int*& F, OpenMesh::SmartFaceHandle*& Queue) {
		for (MyMesh::FaceFaceIter ff = mesh.ff_iter(fh); ff.is_valid(); ++ff) {
			OpenMesh::SmartFaceHandle f = *ff;
			if (F[f.idx()] == 0) {
				//for (const auto& e : f.edges()) std::cout << e.idx() << endl;
				Queue[rear] = f;
				//for (const auto& e : Queue[rear].edges()) std::cout << e.idx() << endl;
				F[f.idx()]++;
				//std::cout << f << "  " << Queue[rear] << endl;
				rear++;
			}
		}
	}

	// 初始化面
	static void triangle_init(MyMesh mesh, vector<vector<double>>& Axis, double* L, int& rear, int*& F, OpenMesh::SmartFaceHandle*& Queue) {
		for (const auto& fh : mesh.faces()) {
			int i = 0;
			double l1 = 0, l2 = 0, l3 = 0;
			for (const auto& e : fh.edges()) {
				if (i == 0) {
					l1 = L[e.idx()];
					Axis[e.v0().idx()][0] = -l1 / 2.0;
					Axis[e.v1().idx()][0] = l1 / 2.0;
				}
				else if (i == 1) l2 = L[e.idx()];
				else {
					l3 = L[e.idx()];
					double theta = acos((l1 * l1 + l3 * l3 - l2 * l2) / (2.0 * l1 * l3));
					Axis[e.v0().idx()][0] = l3 * cos(theta) + (-l1 / 2.0);
					Axis[e.v0().idx()][1] = l3 * sin(theta);
				}
				i++;
			}
			F[fh.idx()]++;
			triangle_queue(mesh, fh, rear, F, Queue);
			break;
		}
		std::cout << "Mesh Init Done!" << endl;
		return;
	}

	// 单点插入
	static void triangle_add(vector<vector<double>>& Axis, int V[], double l[], int x) {
		double iota, theta;
		if ((Axis[V[(x + 2) % 3]][0] - Axis[V[(x + 1) % 3]][0]) == 0) iota = 0.5 * M_PI;
		else iota = atan((Axis[V[(x + 2) % 3]][1] - Axis[V[(x + 1) % 3]][1]) / (Axis[V[(x + 2) % 3]][0] - Axis[V[(x + 1) % 3]][0]));
		// right
		if (Axis[V[(x + 2) % 3]][1] < Axis[V[(x + 1) % 3]][1]) {
			theta = acos((l[(x + 1) % 3] * l[(x + 1) % 3] + l[(x + 2) % 3] * l[(x + 2) % 3] - l[x] * l[x]) / (2.0 * l[(x + 1) % 3] * l[(x + 2) % 3]));
			Axis[V[x]][0] = Axis[V[(x + 2) % 3]][0] + cos(iota - theta) * l[(x + 2) % 3];
			Axis[V[x]][1] = Axis[V[(x + 2) % 3]][1] + sin(iota - theta) * l[(x + 2) % 3];
		}
		// left
		else if (Axis[V[(x + 2) % 3]][1] > Axis[V[(x + 1) % 3]][1]) {
			theta = acos((l[(x + 1) % 3] * l[(x + 1) % 3] + l[x] * l[x] - l[(x + 2) % 3] * l[(x + 2) % 3]) / (2.0 * l[(x + 1) % 3] * l[x]));
			Axis[V[x]][0] = Axis[V[(x + 1) % 3]][0] - cos(M_PI - iota - theta) * l[x];
			Axis[V[x]][1] = Axis[V[(x + 1) % 3]][1] + sin(M_PI - iota - theta) * l[x];
		}
		// up
		else if (Axis[V[(x + 2) % 3]][0] > Axis[V[(x + 1) % 3]][0]) {
			theta = acos((l[(x + 1) % 3] * l[(x + 1) % 3] + l[x] * l[x] - l[(x + 2) % 3] * l[(x + 2) % 3]) / (2.0 * l[(x + 1) % 3] * l[x]));
			Axis[V[x]][0] = Axis[V[(x + 1) % 3]][0] + cos(theta) * l[x];
			Axis[V[x]][1] = Axis[V[(x + 1) % 3]][1] + sin(theta) * l[x];
		}
		// down
		else {
			theta = acos((l[(x + 1) % 3] * l[(x + 1) % 3] + l[(x + 2) % 3] * l[(x + 2) % 3] - l[x] * l[x]) / (2.0 * l[(x + 1) % 3] * l[(x + 2) % 3]));
			Axis[V[x]][0] = Axis[V[(x + 2) % 3]][0] + cos(theta) * l[(x + 2) % 3];
			Axis[V[x]][1] = Axis[V[(x + 2) % 3]][1] - sin(theta) * l[(x + 2) % 3];
		}
		return;
	}

	// 坐标插入
	static void insert_axis(MyMesh mesh, double* L, vector<vector<double>>& Axis) {

		// 面访问指示器 & 面遍历队列 & 队首尾指针
		int* F = new int[mesh.n_faces()]();
		OpenMesh::SmartFaceHandle* Queue_Face = new OpenMesh::SmartFaceHandle[mesh.n_faces() + 3]();
		int Q_front = 0, Q_rear = 0;

		// 初始化
		triangle_init(mesh, Axis, L, Q_rear, F, Queue_Face);

		int x = -1, i = 0;
		int V[3] = { 0 };
		double l[3] = { 0 };
		// 循环插入
		//std::cout << "OuterCycle:" << endl;
		for (; Q_front < Q_rear; Q_front++) {
			x = -1, i = 0;
			std::cout << "InertCycle: Front - " << Q_front << " and  Rear - " << Q_rear << endl << "Face: " << Queue_Face[Q_front].idx() << endl;
			for (MyMesh::FaceEdgeIter fe = mesh.fe_iter(Queue_Face[Q_front]); fe.is_valid(); ++fe) {
				OpenMesh::SmartEdgeHandle e = *fe;
				//std::cout << "Get_VerticeId" << endl;
				V[i] = e.v0().idx();
				//std::cout << "Get_Length" << endl;
				l[i] = L[e.idx()];
				//std::cout << "Get_InsertVertice" << endl;
				if (Axis[V[i]][0] == 0 && Axis[V[i]][1] == 0) x = i;
				i++;
			}
			//std::cout << "Insert : Face " << Queue_Face[Q_front].idx() << endl;
			triangle_queue(mesh, Queue_Face[Q_front], Q_rear, F, Queue_Face);
			//std::cout << "Insert : Vertices " << V[(x + 3) % 3] << endl;
			if (x == -1) continue;
			else triangle_add(Axis, V, l, x);
		}
		return;
	}
};

/*------表面展平 by DeepSeek------*/

class MeshFlattener {
private:
	MyMesh& mesh;
	double* L;
	std::vector<std::vector<double>>& Axis;
	std::vector<bool> visited_faces;
	std::queue<OpenMesh::SmartFaceHandle> face_queue;

	// 新增：记录每个面的法向方向（用于一致性检查）
	std::vector<int> face_orientation;

public:
	MeshFlattener(MyMesh& mesh, double* L, std::vector<std::vector<double>>& Axis)
		: mesh(mesh), L(L), Axis(Axis) {
		visited_faces.resize(mesh.n_faces(), false);
		face_orientation.resize(mesh.n_faces(), 0); // 0=未定义, 1=正向, -1=反向
		Axis.resize(mesh.n_vertices(), std::vector<double>(2, 0.0));
	}

	bool flattenMesh() {
		if (mesh.n_faces() == 0) {
			std::cout << "Error: Mesh has no faces!" << std::endl;
			return false;
		}

		if (!initializeReferenceTriangle()) {
			std::cout << "Error: Failed to initialize reference triangle!" << std::endl;
			return false;
		}

		return processRemainingFaces();
	}

	// 使用函数
	static void flattenMesh(MyMesh mesh, double* L, std::vector<std::vector<double>>& Axis) {
		std::cout << "Starting consistent mesh flattening..." << std::endl;

		MeshFlattener flattener(mesh, L, Axis);

		if (flattener.flattenMesh()) std::cout << "Mesh flattening completed successfully!" << std::endl;
		else std::cout << "Mesh flattening failed or partially completed!" << std::endl;
	}

private:
	bool initializeReferenceTriangle() {
		OpenMesh::SmartFaceHandle first_face = *mesh.faces_begin();

		std::vector<OpenMesh::SmartVertexHandle> vertices;
		for (auto fv_it = mesh.fv_iter(first_face); fv_it.is_valid(); ++fv_it) {
			vertices.push_back(*fv_it);
		}

		if (vertices.size() != 3) return false;

		// 获取三条边的长度
		std::vector<double> lengths;
		for (auto fe_it = mesh.fe_iter(first_face); fe_it.is_valid(); ++fe_it) {
			lengths.push_back(L[fe_it->idx()]);
		}

		if (!isValidTriangle(lengths[0], lengths[1], lengths[2])) {
			return false;
		}

		// 放置第一个三角形
		Axis[vertices[0].idx()][0] = 0.0;
		Axis[vertices[0].idx()][1] = 0.0;

		Axis[vertices[1].idx()][0] = lengths[0];
		Axis[vertices[1].idx()][1] = 0.0;

		double cos_angle = (lengths[0] * lengths[0] + lengths[1] * lengths[1] - lengths[2] * lengths[2]) / (2.0 * lengths[0] * lengths[1]);
		cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
		double sin_angle = std::sqrt(1.0 - cos_angle * cos_angle);

		Axis[vertices[2].idx()][0] = lengths[1] * cos_angle;
		Axis[vertices[2].idx()][1] = lengths[1] * sin_angle;

		// 设置参考面的法向方向为正向
		face_orientation[first_face.idx()] = 1;

		visited_faces[first_face.idx()] = true;
		addNeighborFacesToQueue(first_face);

		std::cout << "Reference triangle " << first_face.idx() << " initialized with orientation +1" << std::endl;
		return true;
	}

	bool processRemainingFaces() {
		int processed_count = 1;

		while (!face_queue.empty()) {
			OpenMesh::SmartFaceHandle current_face = face_queue.front();
			face_queue.pop();

			if (visited_faces[current_face.idx()]) continue;

			if (flattenFaceWithConsistency(current_face)) {
				visited_faces[current_face.idx()] = true;
				processed_count++;
				addNeighborFacesToQueue(current_face);

				std::cout << "Successfully flattened face " << current_face.idx()
					<< " with orientation " << face_orientation[current_face.idx()] << std::endl;
			}
			else {
				std::cout << "Failed to flatten face " << current_face.idx() << std::endl;
				// 将失败的面重新加入队列末尾，等待后续处理
				face_queue.push(current_face);
			}

			// 防止无限循环
			if (face_queue.size() > mesh.n_faces() * 2) {
				std::cout << "Warning: Possible infinite loop detected!" << std::endl;
				break;
			}
		}

		std::cout << "Mesh flattening completed. Processed " << processed_count
			<< " faces." << std::endl;
		return processed_count == mesh.n_faces();
	}

	// 改进的面展平函数，包含一致性检查
	bool flattenFaceWithConsistency(OpenMesh::SmartFaceHandle face) {
		std::vector<OpenMesh::SmartVertexHandle> vertices;
		std::vector<OpenMesh::SmartEdgeHandle> edges;
		std::vector<double> lengths;

		for (auto fv_it = mesh.fv_iter(face); fv_it.is_valid(); ++fv_it) {
			vertices.push_back(*fv_it);
		}
		for (auto fe_it = mesh.fe_iter(face); fe_it.is_valid(); ++fe_it) {
			edges.push_back(*fe_it);
			lengths.push_back(L[fe_it->idx()]);
		}

		if (vertices.size() != 3) return false;

		// 查找未知顶点
		int unknown_vertex_index = -1;
		std::vector<int> known_vertices;

		for (int i = 0; i < 3; i++) {
			if (isVertexPositioned(vertices[i].idx())) {
				known_vertices.push_back(i);
			}
			else {
				unknown_vertex_index = i;
			}
		}

		if (unknown_vertex_index == -1) {
			// 所有顶点都已定位，只进行一致性检查
			face_orientation[face.idx()] = calculateFaceOrientation(vertices);
			return true;
		}

		if (known_vertices.size() < 2) {
			return false; // 需要至少两个已知顶点
		}

		// 获取相邻面的法向信息来指导当前面的方向选择
		int preferred_orientation = getPreferredOrientationFromNeighbors(face);

		return calculateUnknownVertexWithOrientation(vertices, lengths, unknown_vertex_index, face, preferred_orientation);
	}

	// 从相邻面获取推荐的法向方向
	int getPreferredOrientationFromNeighbors(OpenMesh::SmartFaceHandle face) {
		int positive_count = 0;
		int negative_count = 0;

		for (auto ff_it = mesh.ff_iter(face); ff_it.is_valid(); ++ff_it) {
			OpenMesh::SmartFaceHandle neighbor = *ff_it;
			if (visited_faces[neighbor.idx()] && face_orientation[neighbor.idx()] != 0) {
				if (face_orientation[neighbor.idx()] > 0) {
					positive_count++;
				}
				else {
					negative_count++;
				}
			}
		}

		// 如果大多数相邻面是正向，推荐正向；否则推荐反向
		if (positive_count >= negative_count) {
			return 1;
		}
		else {
			return -1;
		}
	}

	// 带方向控制的顶点位置计算
	bool calculateUnknownVertexWithOrientation(const std::vector<OpenMesh::SmartVertexHandle>& vertices,
		const std::vector<double>& lengths,
		int unknown_idx,
		OpenMesh::SmartFaceHandle face,
		int preferred_orientation) {
		int known_idx1 = (unknown_idx + 1) % 3;
		int known_idx2 = (unknown_idx + 2) % 3;

		OpenMesh::SmartVertexHandle v_unknown = vertices[unknown_idx];
		OpenMesh::SmartVertexHandle v_known1 = vertices[known_idx1];
		OpenMesh::SmartVertexHandle v_known2 = vertices[known_idx2];

		double a = lengths[unknown_idx];
		double b = lengths[known_idx1];
		double c = lengths[known_idx2];

		double x1 = Axis[v_known1.idx()][0], y1 = Axis[v_known1.idx()][1];
		double x2 = Axis[v_known2.idx()][0], y2 = Axis[v_known2.idx()][1];

		double dx = x2 - x1, dy = y2 - y1;
		double base_length = std::sqrt(dx * dx + dy * dy);

		if (std::abs(base_length - a) > 1e-5) {
			std::cout << "Warning: Base length mismatch in face " << face.idx()
				<< ": expected " << a << ", got " << base_length << std::endl;
		}

		// 使用余弦定理
		double cos_angle = (b * b + c * c - a * a) / (2.0 * b * c);
		cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
		double angle = std::acos(cos_angle);

		double base_angle = std::atan2(dy, dx);

		// 尝试两个可能的解
		std::vector<std::vector<double>> candidate_positions(2);

		// 解1：左侧
		candidate_positions[0].resize(2);
		candidate_positions[0][0] = x1 + b * std::cos(base_angle + angle);
		candidate_positions[0][1] = y1 + b * std::sin(base_angle + angle);

		// 解2：右侧  
		candidate_positions[1].resize(2);
		candidate_positions[1][0] = x1 + b * std::cos(base_angle - angle);
		candidate_positions[1][1] = y1 + b * std::sin(base_angle - angle);

		// 选择最佳解
		int best_solution = selectBestSolution(face, v_unknown, candidate_positions, preferred_orientation);

		if (best_solution == -1) {
			return false;
		}

		// 应用选择的解
		Axis[v_unknown.idx()][0] = candidate_positions[best_solution][0];
		Axis[v_unknown.idx()][1] = candidate_positions[best_solution][1];

		// 计算并存储面的法向方向
		face_orientation[face.idx()] = calculateFaceOrientation(vertices);

		// 验证与相邻面的一致性
		if (!verifyNeighborConsistency(face)) {
			std::cout << "Warning: Face " << face.idx() << " may have consistency issues with neighbors" << std::endl;
		}

		return true;
	}

	// 选择最佳解
	int selectBestSolution(OpenMesh::SmartFaceHandle face,
		OpenMesh::SmartVertexHandle unknown_vertex,
		const std::vector<std::vector<double>>& candidates,
		int preferred_orientation) {
		// 检查与相邻面的几何一致性
		double best_score = -1.0;
		int best_solution = -1;

		for (int i = 0; i < 2; i++) {
			double score = calculateSolutionScore(face, unknown_vertex, candidates[i], preferred_orientation);

			if (score > best_score) {
				best_score = score;
				best_solution = i;
			}
		}

		if (best_solution == 0) {
			std::cout << "Selected LEFT solution for face " << face.idx() << " (score: " << best_score << ")" << std::endl;
		}
		else {
			std::cout << "Selected RIGHT solution for face " << face.idx() << " (score: " << best_score << ")" << std::endl;
		}

		return best_solution;
	}

	// 计算解的评分
	double calculateSolutionScore(OpenMesh::SmartFaceHandle face,
		OpenMesh::SmartVertexHandle unknown_vertex,
		const std::vector<double>& candidate_pos,
		int preferred_orientation) {
		double score = 0.0;

		// 临时保存候选位置
		std::vector<double> original_pos = Axis[unknown_vertex.idx()];
		Axis[unknown_vertex.idx()] = candidate_pos;

		// 1. 方向一致性（40%权重）
		std::vector<OpenMesh::SmartVertexHandle> vertices;
		for (auto fv_it = mesh.fv_iter(face); fv_it.is_valid(); ++fv_it) {
			vertices.push_back(*fv_it);
		}
		int calculated_orientation = calculateFaceOrientation(vertices);
		if (calculated_orientation == preferred_orientation) {
			score += 0.4;
		}

		// 2. 与相邻面的距离一致性（40%权重）
		score += 0.4 * calculateNeighborDistanceConsistency(face);

		// 3. 避免自相交（20%权重）
		if (!checkSelfIntersection(face)) {
			score += 0.2;
		}

		// 恢复原始位置
		Axis[unknown_vertex.idx()] = original_pos;

		return score;
	}

	// 计算面的法向方向（基于2D叉积）
	int calculateFaceOrientation(const std::vector<OpenMesh::SmartVertexHandle>& vertices) {
		if (vertices.size() != 3) return 0;

		double x1 = Axis[vertices[0].idx()][0], y1 = Axis[vertices[0].idx()][1];
		double x2 = Axis[vertices[1].idx()][0], y2 = Axis[vertices[1].idx()][1];
		double x3 = Axis[vertices[2].idx()][0], y3 = Axis[vertices[2].idx()][1];

		double cross_product = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);

		if (cross_product > 0) return 1;   // 逆时针（正向）
		else if (cross_product < 0) return -1; // 顺时针（反向）
		else return 0; // 共线
	}

	// 计算与相邻面的距离一致性
	double calculateNeighborDistanceConsistency(OpenMesh::SmartFaceHandle face) {
		double total_consistency = 0.0;
		int count = 0;

		for (auto ff_it = mesh.ff_iter(face); ff_it.is_valid(); ++ff_it) {
			OpenMesh::SmartFaceHandle neighbor = *ff_it;
			if (visited_faces[neighbor.idx()]) {
				// 计算共享边的长度一致性
				double consistency = calculateEdgeLengthConsistency(face, neighbor);
				total_consistency += consistency;
				count++;
			}
		}

		return count > 0 ? total_consistency / count : 1.0;
	}

	// 计算共享边长度一致性
	double calculateEdgeLengthConsistency(OpenMesh::SmartFaceHandle face1, OpenMesh::SmartFaceHandle face2) {
		// 找到共享边
		for (auto fe1_it = mesh.fe_iter(face1); fe1_it.is_valid(); ++fe1_it) {
			for (auto fe2_it = mesh.fe_iter(face2); fe2_it.is_valid(); ++fe2_it) {
				if (fe1_it->idx() == fe2_it->idx()) {
					double expected_length = L[fe1_it->idx()];
					// 从两个面的顶点位置计算实际长度
					auto vertices = getEdgeVertices(*fe1_it);
					double actual_length = calculateDistance(vertices.first, vertices.second);

					double error = std::abs(actual_length - expected_length) / expected_length;
					return std::max(0.0, 1.0 - error);
				}
			}
		}
		return 0.0;
	}

	// 检查自相交
	bool checkSelfIntersection(OpenMesh::SmartFaceHandle face) {
		// 简化的自相交检查
		// 在实际应用中可以实现更复杂的几何检查
		return false;
	}

	// 验证与相邻面的一致性
	bool verifyNeighborConsistency(OpenMesh::SmartFaceHandle face) {
		for (auto ff_it = mesh.ff_iter(face); ff_it.is_valid(); ++ff_it) {
			OpenMesh::SmartFaceHandle neighbor = *ff_it;
			if (visited_faces[neighbor.idx()]) {
				if (!checkFacePairConsistency(face, neighbor)) {
					return false;
				}
			}
		}
		return true;
	}

	bool checkFacePairConsistency(OpenMesh::SmartFaceHandle face1, OpenMesh::SmartFaceHandle face2) {
		// 检查共享边的长度一致性
		for (auto fe1_it = mesh.fe_iter(face1); fe1_it.is_valid(); ++fe1_it) {
			for (auto fe2_it = mesh.fe_iter(face2); fe2_it.is_valid(); ++fe2_it) {
				if (fe1_it->idx() == fe2_it->idx()) {
					double expected_length = L[fe1_it->idx()];
					auto vertices = getEdgeVertices(*fe1_it);
					double actual_length = calculateDistance(vertices.first, vertices.second);

					if (std::abs(actual_length - expected_length) > 1e-4) {
						std::cout << "Inconsistency detected between face " << face1.idx()
							<< " and " << face2.idx() << " on edge " << fe1_it->idx()
							<< ": expected " << expected_length << ", got " << actual_length << std::endl;
						return false;
					}
				}
			}
		}
		return true;
	}

	// 辅助函数
	std::pair<OpenMesh::SmartVertexHandle, OpenMesh::SmartVertexHandle> getEdgeVertices(OpenMesh::SmartEdgeHandle edge) {
		return std::make_pair(edge.v0(), edge.v1());
	}

	double calculateDistance(OpenMesh::SmartVertexHandle v1, OpenMesh::SmartVertexHandle v2) {
		double dx = Axis[v1.idx()][0] - Axis[v2.idx()][0];
		double dy = Axis[v1.idx()][1] - Axis[v2.idx()][1];
		return std::sqrt(dx * dx + dy * dy);
	}

	void addNeighborFacesToQueue(OpenMesh::SmartFaceHandle face) {
		for (auto ff_it = mesh.ff_iter(face); ff_it.is_valid(); ++ff_it) {
			OpenMesh::SmartFaceHandle neighbor = *ff_it;
			if (!visited_faces[neighbor.idx()]) {
				face_queue.push(neighbor);
			}
		}
	}

	bool isVertexPositioned(int vertex_id) {
		return !(std::abs(Axis[vertex_id][0]) < 1e-10 && std::abs(Axis[vertex_id][1]) < 1e-10);
	}

	bool isValidTriangle(double a, double b, double c) {
		return (a + b > c) && (a + c > b) && (b + c > a);
	}
};

/*------------主体函数------------*/

vector<vector<double>> ricciflow(MyMesh mesh) {
	
	// 顶点半径
	double* Gama = new double[mesh.n_vertices()]();

	// 双曲距离 & 距离 & 非欧计数
	double* I = new double[mesh.n_edges()]();
	double* L = new double[mesh.n_edges()]();
	int* F = new int[mesh.n_faces()]();

	// 曲率夹角 & 偏导
	vector<vector<double>> Theta(mesh.n_faces(), vector<double>(3, 0));
	vector<vector<double>> Del(mesh.n_faces(), vector<double>(3, 0));

	// 高斯曲率 & 自定曲率
	Mat K = Mat_<double>::zeros(mesh.n_vertices(), 1);
	double* K_ = new double[mesh.n_vertices()]();

	// 黑塞矩阵
	Mat Delta = Mat_<double>::zeros(mesh.n_vertices(), mesh.n_vertices());

	// 结果矩阵
	Mat U = Mat_<double>::zeros(mesh.n_vertices(), 1);

	// 初始化
	get_gama(mesh, Gama, U);
	//icheck(Gama, mesh.n_vertices());
	
	// 边界总长
	double boundarylength = get_distance(mesh, Gama, I, L, F);
	//icheck(I, mesh.n_edges());//icheck(L, mesh.n_edges());//icheck(F, mesh.n_faces(), 50);
	get_theta(mesh, L, F, Theta);
	//icheck(Theta, mesh.n_faces(), 3);
	get_round(mesh, Gama, Del);
	//icheck(Del, mesh.n_faces(), 3);
	get_gauss(mesh, Theta, K, K_, boundarylength, BOUNDARYFREE);
	//icheck(K, mesh.n_vertices(), 1); icheck(K_, mesh.n_vertices(), 25);

	// 循环计数 & 曲率差值
	int i = 0;
	double k = 0, k_ = 0;
	
	if (ENGINE == "matlab") {
		// Matlab初始化
		if (mclInitializeApplication(NULL, 0)) cout << "Initialize Matlab." << endl;
		// 动态库初始化
		if (METHOD == "lsqminnorm") {
			if (lsqminnormsolverInitialize()) cout << "Initialize Lsqminnormsolver." << endl;
		}
		else if (METHOD == "svd") {
			if (svdsolverInitialize()) cout << "Initialize SVDsolver." << endl;
		}
		else if (METHOD == "pcg")
		{
			if (pcgsolverInitialize()) cout << "Initialize PCGsolver." << endl;
		}
		else {
			if (matricsolverInitialize()) cout << "Initialize Matricsolver." << endl;
		}
	}

	// 循环体
	while (i < TIMES) {
		get_delta(mesh, Del, Delta);
		//if(i == 0) icheck(Delta, mesh.n_vertices(), mesh.n_vertices(), 45);
		U += solve_mu(Delta, K, mesh.n_vertices());
		//icheck(U, mesh.n_vertices(), 1);
		change_gama(Gama, U, mesh.n_vertices());
		//icheck(Gama, mesh.n_vertices());
		change_distance(mesh, Gama, I, L);
		//icheck(L, mesh.n_edges());
		change_theta(mesh, L, Theta);
		//icheck(Theta, mesh.n_faces(), 3);
		change_round(mesh, Gama, L, Del);
		//icheck(Del, mesh.n_faces(), 3);
		k = change_gauss(mesh, Theta, K, K_, BOUNDARYFREE);
		//icheck(K, mesh.n_vertices(), 1);

		std::cout << endl << "Round " << ++i << " : " << " max( |Ki_ - Ki| ) = " << k << endl << endl;
		if (k < EPSILON) {
			std::cout << i << " Times Converged!	Max(| Ki_- Ki |) < " << EPSILON << endl;
			break;
		}
		if (k == k_)
		{
			std::cout << "Unable to Continue Converging!	Max(| Ki_- Ki |) = " << k << endl;
			break;
		}
		k_ = k;
	}

	if (ENGINE == "matlab") {
		// 动态库关闭
		if (METHOD == "lsqminnorm") {
			lsqminnormsolverTerminate();
		}
		else if (METHOD == "svd") {
			svdsolverTerminate();
		}
		else if (METHOD == "pcg")
		{
			pcgsolverTerminate();
		}
		else {
			matricsolverTerminate();
		}
		// Matlab关闭
		mclTerminateApplication();
	}

	// 纹理坐标
	vector<vector<double>> Axis(mesh.n_vertices(), vector<double>(2, 0));
	//icheck(L, mesh.n_edges());
	MeshFlattener::flattenMesh(mesh, L, Axis);
	//icheck(Axis, mesh.n_vertices(), 2);
	return Axis;
}