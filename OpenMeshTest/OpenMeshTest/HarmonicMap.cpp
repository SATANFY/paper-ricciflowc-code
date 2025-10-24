#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

using namespace std;
using namespace cv;

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

// 区分内外点集
int get_vertices(MyMesh mesh, OpenMesh::SmartVertexHandle*& IV, vector<vector<double>>& OV)
{
	int i = 0, j = 0;
	int* BV = new int[mesh.n_vertices()]();
	double l = 0, bl = 0;
	double* BE = new double[mesh.n_edges()]();
	for (const auto& v : mesh.vertices())
	{
		if (mesh.is_boundary(v) && i == 0)
		{
			BV[i] = v.idx();
			for (MyMesh::VertexIHalfedgeIter h = mesh.vih_iter(v); h.is_valid(); ++h)
			{
				MyMesh::HalfedgeHandle he = *h;
				if (mesh.is_boundary(he))
				{
					BE[i] = mesh.calc_edge_length(he);
					bl += BE[i++];
					while ((int)mesh.from_vertex_handle(he).idx() != BV[0])
					{
						BV[i] = mesh.from_vertex_handle(he).idx();
						he = mesh.prev_halfedge_handle(he);
						BE[i] = mesh.calc_edge_length(he);
						bl += BE[i++];
					} 
					cout << "Get_Boundary_Vertice." << endl;
					break;
				}
			}
		}
		else if (!mesh.is_boundary(v))
		{
			IV[j++] = v;
		}
	}
	cout << "Get_Interior_Vertice." << endl;
	// 边界点映射
	OV[BV[0]][0] = cos(2 * M_PI);	OV[BV[0]][1] = sin(2 * M_PI);
	for (int k = 1; k < i; k++)
	{
		l += BE[k - 1];
		double theta = 2 * M_PI * l / bl;
		OV[BV[k]][0] = cos(theta);
		OV[BV[k]][1] = sin(theta);
	}
	cout << "Get_Out_Boundary_Vertice." << endl;
	return j;
}

// 求边余切权重
void get_omegas(MyMesh mesh, double*& OMEGA)
{
	for (const auto& eh : mesh.edges())
	{
		MyMesh::HalfedgeHandle hh = eh.halfedge();
		if (mesh.is_boundary(eh))
		{
			if (!mesh.is_boundary(hh)) {
				hh = mesh.opposite_halfedge_handle(hh);
			}
			hh = mesh.prev_halfedge_handle(hh);
			OMEGA[eh.idx()] = 0.5 * (1.0 / tan(mesh.calc_sector_angle(hh)));
		}
		else
		{
			MyMesh::HalfedgeHandle op_hh = mesh.opposite_halfedge_handle(hh);
			hh = mesh.prev_halfedge_handle(hh);
			op_hh = mesh.prev_halfedge_handle(op_hh);
			OMEGA[eh.idx()] = 0.5 * (1.0 / tan(mesh.calc_sector_angle(hh)) + 1.0 / tan(mesh.calc_sector_angle(op_hh)));
		}
	}
	cout << "Get_Edge_Omega." << endl;
	return;
}

// 构造方程矩阵
void get_matrix(MyMesh mesh, int n_iv, OpenMesh::SmartVertexHandle* IV, vector<vector<double>> OV, double* OMEGA, Mat& A, Mat& B) {
	for (int i = 0; i < n_iv; i++)
	{
		for (MyMesh::VertexEdgeIter ve = mesh.ve_iter(IV[i]); ve.is_valid(); ++ve)
		{
			int j = 0;
			OpenMesh::SmartEdgeHandle e = *ve;
			if (mesh.is_boundary(e.v0())) {
				B.at<double>(i, 0) += OMEGA[e.idx()] * OV[e.v0().idx()][0];
				B.at<double>(i, 1) += OMEGA[e.idx()] * OV[e.v0().idx()][1];
			}
			else {
				for (j = 0; j < n_iv; j++)
					if (IV[j].idx() == e.v0().idx())
						break;
				if (i == j)
					A.at<double>(i, j) += OMEGA[e.idx()];
				else
					A.at<double>(i, j) -= OMEGA[e.idx()];
			}
			if (mesh.is_boundary(e.v1())) {
				B.at<double>(i, 0) += OMEGA[e.idx()] * OV[e.v1().idx()][0];
				B.at<double>(i, 1) += OMEGA[e.idx()] * OV[e.v1().idx()][1];
			}
			else {
				for (j = 0; j < n_iv; j++) 
					if (IV[j].idx() == e.v1().idx())
						break;
				if (i == j)
					A.at<double>(i, j) += OMEGA[e.idx()];
				else
					A.at<double>(i, j) -= OMEGA[e.idx()];
			}
		}
	}
	cout << "Get_Matrix_Parameter." << endl;
	return;
}

// 求解二元方程
Mat solve_matrix(Mat A, Mat B, int n_iv)
{
	Mat X = Mat_<double>::zeros(n_iv, 2);
	cout << "Start_Solve_Matrix ..." << endl;
	bool flag = solve(A, B, X, DECOMP_NORMAL);
	cout << "End_Solve_Matrix." << endl;
	if (flag == false) cout << "Failed to Find the Result." << endl;
	else cout << "Find the Result." << endl;
	return X;
}


vector<vector<double>> harmonicmap(MyMesh mesh) {
	
	// 纹理坐标
	vector<vector<double>> OV(mesh.n_vertices(), vector<double>(2, 0));

	// 内部点集
	OpenMesh::SmartVertexHandle* IV = new OpenMesh::SmartVertexHandle[mesh.n_vertices()]();
	
	// 余切权重
	double* OMEGA = new double[mesh.n_edges()]();

	// 外部坐标
	int n_iv = get_vertices(mesh, IV, OV);

	// 求边权重
	get_omegas(mesh, OMEGA);

	// 矩阵向量
	Mat A = Mat_<double>::zeros(n_iv, n_iv);
	Mat B = Mat_<double>::zeros(n_iv, 2);

	// 构建方程
	get_matrix(mesh, n_iv, IV, OV, OMEGA, A, B);
			
	// 内部坐标
	B = solve_matrix(A, B, n_iv);
	for (int i = 0; i < n_iv; i++)
	{
		OV[IV[i].idx()][0] = B.at<double>(i, 0);
		OV[IV[i].idx()][1] = B.at<double>(i, 1);
	}

	return OV;
}