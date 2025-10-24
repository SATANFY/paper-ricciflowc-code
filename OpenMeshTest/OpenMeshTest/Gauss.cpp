#include <iostream>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

using namespace std;

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;
typedef OpenMesh::SmartVertexHandle SmartVH;

double gauss(MyMesh mesh,SmartVH vh) {
	double flag = 1.0, t_angle = 0.0;
	cout << endl;
	for (MyMesh::VertexIHalfedgeIter i = mesh.vih_iter(vh); i.is_valid(); ++i)
	{
		MyMesh::HalfedgeHandle he = *i;
		if (!mesh.is_boundary(he))
		{
			cout << "б╧" << he.idx() << " = " << mesh.calc_sector_angle(he) / (2 * 3.14) * 360 << "бу  " << mesh.calc_sector_angle(he) << "   ";
			t_angle += mesh.calc_sector_angle(he);
		}
		else flag = -1.0;
	}
	cout << flag << endl;
	return flag * t_angle;
}