#include <iostream>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include "Gauss.h"
#include "HarmonicMap.h"
#include "RicciFlow.h"

using namespace std;
using namespace OpenMesh::Iterators;

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

MyMesh mesh;

// vertice coordinate
void output_vertice(MyMesh mesh) {
	cout << endl << "Vertice Coordinate: " << endl;
	for (const auto& vh : mesh.vertices())
	{
		double K;
		cout << endl << "V" << vh.idx() << " ( " << mesh.point(vh) << " ) " << endl;
		K = gauss(mesh, vh);
		K = (K < 0) ? M_PI + K : 2 * M_PI - K;
		cout << endl << "K" << vh.idx() << " = " << K << endl;
	}
}

// edge length
void output_edge(MyMesh mesh) {
	cout << endl << "Edge Length:" << endl;
	for (const auto& eh : mesh.edges())
	{
		cout << endl << "E" << eh.idx() << " : " << mesh.calc_edge_length(eh) << endl;
	}
}

// halfedge angle
void output_angle(MyMesh mesh) {
	cout << endl << "Angle:" << endl;
	for (const auto& hh : mesh.halfedges())
	{
		cout << endl << "��" << hh.idx() << " : " << mesh.calc_sector_angle(hh) / (2 * M_PI) * 360 << "��" << endl;
	}
}

// write obj
void output_obj(MyMesh outmesh, vector<vector<double>> texVertice, string texName, string fileName) {
	
	// ��ȡ��������
	OpenMesh::VPropHandleT<OpenMesh::Vec2f> texCoords;
	if (!outmesh.get_property_handle(texCoords, "texCoords"))
	{
		outmesh.add_property(texCoords, "texCoords");
	}
	
	// д���ļ�
	ofstream fout(fileName);
	if (!fout.is_open()) 
	{
		cout << "Filed to Open." << endl;
	}
	
	// ������Ϣ
	fout << "mtllib Texture.mtl\nusemtl " + texName + "\n";
	
	// �����꼰��������
	cout << "Writing Vertices & TextureCoodrds ..." << endl;
	for (const  auto& vh : outmesh.vertices())
	{
		MyMesh::Point v = outmesh.point(vh);
		fout << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
		fout << "vt " << texVertice[vh.idx()][0] << " " << texVertice[vh.idx()][1] << "\n";
	}

	// ������
	cout << "Writing Faces ..." << endl;
	for (const auto& fh : outmesh.faces())
	{
		fout << "f ";
		for (MyMesh::FaceVertexIter fv = outmesh.fv_iter(fh); fv.is_valid(); ++fv)
		{
			int i = (fv->idx()) + 1;
			fout << i << "/" << i << " ";
		}
		fout << "\n";
	}

	// �ر��ļ�
	cout << "Write_Obj_File." << endl;
	fout.close();

	// �Ƴ���������
	outmesh.remove_property(texCoords);
	
	return;
}

// harmonic map
void output_harmonic(MyMesh mesh) {
	// get texCoords
	vector<vector<double>> VT(mesh.n_vertices(), vector<double>(2, 0));
	VT = harmonicmap(mesh);

	// output obj
	output_obj(mesh, VT, "Square", "D:\\Yu\\Study\\Yunnan University of Finance and Economics\\202507 TriangleMesh\\File\\HarmonicMap\\result\\result.obj");
}

// ricci flow
void output_ricciflow(MyMesh mesh) {
	// get texCoords
	vector<vector<double>> VT(mesh.n_vertices(), vector<double>(2, 0));
	VT = ricciflow(mesh);

	// output obj
	output_obj(mesh, VT, "Square", "D:\\Yu\\Study\\Yunnan University of Finance and Economics\\202507 TriangleMesh\\File\\RicciFlow\\result\\result.obj");
}

int main() {
	// read mesh # testeigen.obj - v10 # alex300.obj - v300 # alex6k.obj - v6000 #
	if (!OpenMesh::IO::read_mesh(mesh, "D:\\Yu\\Study\\Yunnan University of Finance and Economics\\202507 TriangleMesh\\Model\\alex6k.obj"))
	{
		cout << "Read Mesh Error." << endl;
		exit(1);
	}
	output_ricciflow(mesh);
}
