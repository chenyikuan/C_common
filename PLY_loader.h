#ifndef CYK_PLY_LOADDER_H_
#define CYK_PLY_LOADDER_H_

#include "windows.h"

#include <GLMatrixStack.h>
#include <GLFrame.h>
#include <GLFrustum.h>
#include <GLGeometryTransform.h>

#define NDEBUG
#define FREEGLUT_STATIC
#include <GL/glut.h>
//#include <GL/glu.h>
//#include <GL/gl.h>
#include <vector>
#include <iostream>

//struct SModelData
//{
//	std::vector <float> vecFaceTriangles; // = face * 9
//	std::vector <float> vecFaceTriangleColors; // = face * 9
//	std::vector <float> vecNormals; // = face * 9
//	int iTotalConnectedTriangles;
//};

class ply_loader
{
public:
	ply_loader();
	~ply_loader();
	//int LoadModel(char *filename);
	int LoadModel(const std::string& filename);
	//void Draw();
	void update_mdbatch(GLBatch& batch);
	void clear();

private:
	float* mp_vertexXYZ;
	float* mp_vertexNorm;
	float* mp_vertexRGB;

	float* triangles;
	float* triangle_norms;
	float* colors;

	int m_totalConnectedQuads;
	int m_totalConnectedPoints;
	int m_totalFaces;
	//SModelData m_ModelData;
};

#endif