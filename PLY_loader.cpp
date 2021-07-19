#include "PLY_loader.h"
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

ply_loader::ply_loader()
{
	this->m_totalConnectedQuads = 0;
	this->m_totalConnectedPoints = 0;
	//m_ModelData.iTotalConnectedTriangles = 0;
}

ply_loader::~ply_loader()
{
	cout << "Freeing space ... ";
	free(mp_vertexXYZ);
	free(mp_vertexNorm);
	free(mp_vertexRGB);
	free(triangles);
	free(triangle_norms);
	free(colors);
	cout << "OK." << endl;
}

void ply_loader::clear()
{
	this->m_totalConnectedQuads = 0;
	this->m_totalConnectedPoints = 0;
	this->m_totalFaces = 0;
	free(mp_vertexXYZ);
	free(mp_vertexNorm);
	free(mp_vertexRGB);
	free(triangles);
	free(triangle_norms);
	free(colors);
}

//int ply_loader::LoadModel(char* filename)
//{
//	printf("Loading %s ...\n", filename);
//	char* pch = strstr(filename, ".ply");
//
//	if (pch != NULL)
//	{
//		FILE* file = fopen(filename, "r");
//		if (!file)
//		{
//			printf("load PLY file %s failed\n", filename);
//			return false;
//		}
//		fseek(file, 0, SEEK_END);
//		long fileSize = ftell(file);
//
//		try
//		{
//			mp_vertexXYZ = (float*)malloc(ftell(file));
//			mp_vertexNorm = (float*)malloc(ftell(file));
//			mp_vertexRGB = (float*)malloc(ftell(file));
//		}
//		catch (char*)
//		{
//			return -1;
//		}
//		if (mp_vertexXYZ == NULL) return -1;
//		if (mp_vertexNorm == NULL) return -2;
//		if (mp_vertexRGB == NULL) return -3;
//		fseek(file, 0, SEEK_SET);
//
//		if (file)
//		{
//			int i = 0;
//			int temp = 0;
//			int quads_index = 0;
//			int triangle_index = 0;
//			int normal_index = 0;
//			int colorIndex = 0;
//			char buffer[1000];
//
//
//			fgets(buffer, 300, file);			// ply
//
//
//			// READ HEADER
//			// -----------------
//
//			// Find number of vertexes
//			while (strncmp("element vertex", buffer, strlen("element vertex")) != 0)
//			{
//				fgets(buffer, 300, file);			// format
//			}
//			strcpy(buffer, buffer + strlen("element vertex"));
//			sscanf(buffer, "%i", &this->m_totalConnectedPoints);
//
//
//			// Find number of vertexes
//			fseek(file, 0, SEEK_SET);
//			while (strncmp("element face", buffer, strlen("element face")) != 0)
//			{
//				fgets(buffer, 300, file);			// format
//			}
//			strcpy(buffer, buffer + strlen("element face"));
//			sscanf(buffer, "%i", &this->m_totalFaces);
//
//
//			// go to end_header
//			while (strncmp("end_header", buffer, strlen("end_header")) != 0)
//			{
//				fgets(buffer, 300, file);			// format
//			}
//
//			//----------------------
//
//
//			// read vertices
//			i = 0;
//			for (int iterator = 0; iterator < this->m_totalConnectedPoints; iterator++)//3488
//			{
//				cout << m_totalConnectedPoints << endl;
//				char tmp[1];
//				fgets(buffer, 300, file);
//				cout << buffer << endl;
//
//				sscanf(buffer, "%f %f %f %f %f %f %f %f %f", mp_vertexXYZ[i], mp_vertexXYZ[i + 1], mp_vertexXYZ[i + 2],
//					mp_vertexNorm[i], mp_vertexNorm[i + 1], mp_vertexNorm[i + 2],
//					mp_vertexRGB[i], mp_vertexRGB[i + 1], mp_vertexRGB[i + 2]);
//				i += 3;
//			}
//
//			// read faces
//			i = 0;
//			for (int iterator = 0; iterator < this->m_totalFaces; iterator++)//6920
//			{
//				fgets(buffer, 300, file);
//
//				if (buffer[0] == '3')
//				{
//					int vertex1 = 0, vertex2 = 0, vertex3 = 0;
//					buffer[0] = ' ';
//					sscanf(buffer, "%i%i%i", &vertex1, &vertex2, &vertex3);//number of vertex eg:5,7,6
//
//
//					m_ModelData.vecFaceTriangles.push_back(mp_vertexXYZ[3 * vertex1]);
//					m_ModelData.vecFaceTriangles.push_back(mp_vertexXYZ[3 * vertex1 + 1]);
//					m_ModelData.vecFaceTriangles.push_back(mp_vertexXYZ[3 * vertex1 + 2]);
//					m_ModelData.vecFaceTriangles.push_back(mp_vertexXYZ[3 * vertex2]);
//					m_ModelData.vecFaceTriangles.push_back(mp_vertexXYZ[3 * vertex2 + 1]);
//					m_ModelData.vecFaceTriangles.push_back(mp_vertexXYZ[3 * vertex2 + 2]);
//					m_ModelData.vecFaceTriangles.push_back(mp_vertexXYZ[3 * vertex3]);
//					m_ModelData.vecFaceTriangles.push_back(mp_vertexXYZ[3 * vertex3 + 1]);
//					m_ModelData.vecFaceTriangles.push_back(mp_vertexXYZ[3 * vertex3 + 2]);
//
//
//					m_ModelData.vecFaceTriangleColors.push_back(mp_vertexRGB[3 * vertex1] / 255.0f);
//					m_ModelData.vecFaceTriangleColors.push_back(mp_vertexRGB[3 * vertex1 + 1] / 255.0f);
//					m_ModelData.vecFaceTriangleColors.push_back(mp_vertexRGB[3 * vertex1 + 2] / 255.0f);
//					m_ModelData.vecFaceTriangleColors.push_back(mp_vertexRGB[3 * vertex2] / 255.0f);
//					m_ModelData.vecFaceTriangleColors.push_back(mp_vertexRGB[3 * vertex2 + 1] / 255.0f);
//					m_ModelData.vecFaceTriangleColors.push_back(mp_vertexRGB[3 * vertex2 + 2] / 255.0f);
//					m_ModelData.vecFaceTriangleColors.push_back(mp_vertexRGB[3 * vertex3] / 255.0f);
//					m_ModelData.vecFaceTriangleColors.push_back(mp_vertexRGB[3 * vertex3 + 1] / 255.0f);
//					m_ModelData.vecFaceTriangleColors.push_back(mp_vertexRGB[3 * vertex3 + 2] / 255.0f);
//
//					m_ModelData.vecNormals.push_back(mp_vertexNorm[3 * vertex1]);
//					m_ModelData.vecNormals.push_back(mp_vertexNorm[3 * vertex1 + 1]);
//					m_ModelData.vecNormals.push_back(mp_vertexNorm[3 * vertex1 + 2]);
//					m_ModelData.vecNormals.push_back(mp_vertexNorm[3 * vertex2]);
//					m_ModelData.vecNormals.push_back(mp_vertexNorm[3 * vertex2 + 1]);
//					m_ModelData.vecNormals.push_back(mp_vertexNorm[3 * vertex2 + 2]);
//					m_ModelData.vecNormals.push_back(mp_vertexNorm[3 * vertex3]);
//					m_ModelData.vecNormals.push_back(mp_vertexNorm[3 * vertex3 + 1]);
//					m_ModelData.vecNormals.push_back(mp_vertexNorm[3 * vertex3 + 2]);
//
//					triangle_index += 9;
//					m_ModelData.iTotalConnectedTriangles += 3;
//				}
//
//
//				i += 3;
//			}
//
//			fclose(file);
//			printf("%s Loaded!\n", filename);
//
//		}
//
//		else
//		{
//			printf("File can't be opened\n");
//		}
//	}
//	else
//	{
//		printf("File does not have a .PLY extension. ");
//	}
//
//	return 0;
//}

int ply_loader::LoadModel(const std::string& filename_)
{
	string filename = filename_;
	cout << "Loading " << filename << " ..." << endl;
	string ext = filename.substr(filename.length() - 4, filename.length() - 1);
	cout << ext << endl;

	if (ext == ".ply")
	{
		//FILE* file = fopen(filename, "r");
		ifstream file(filename);
		if (!file.is_open())
		{
			cout << "load PLY file " + filename + " failed" << endl;
			return false;
		}
		streampos pos = file.tellg();

		if (file)
		{
			int i = 0;
			int temp = 0;
			int quads_index = 0;
			int triangle_index = 0;
			int normal_index = 0;
			int colorIndex = 0;
			//char buffer[1000];
			string tmp_line;
			stringstream ss;


			// READ HEADER
			// -----------------
			int info_lines_num = 0;
			do
			{
				getline(file, tmp_line);
				info_lines_num++;
			} while (strncmp("end_header", tmp_line.c_str(), strlen("end_header")) != 0);
			cout << "Info lines num: " << info_lines_num << endl;
			// Find number of vertexes
			temp = 0;
			file.seekg(pos);
			do
			{
				getline(file, tmp_line);
				if (++temp == info_lines_num)
					break;
			} while (strncmp("element vertex", tmp_line.c_str(), strlen("element vertex")) != 0);
			if (temp == info_lines_num)
				tmp_line = "0";
			else
				tmp_line = tmp_line.substr(14, tmp_line.length() - 1);
			this->m_totalConnectedPoints = atoi(tmp_line.c_str());
			cout << "element vertex: " << this->m_totalConnectedPoints <<endl;


			// Find number of vertexes
			//fseek(file, 0, SEEK_SET);
			//while (strncmp("element face", buffer, strlen("element face")) != 0)
			//{
			//	fgets(buffer, 300, file);			// format
			//}
			//strcpy(buffer, buffer + strlen("element face"));
			//sscanf(buffer, "%i", &this->m_totalFaces);
			temp = 0;
			file.seekg(pos);
			do
			{
				getline(file, tmp_line);
				if (++temp == info_lines_num)
					break;
			} while (strncmp("element face", tmp_line.c_str(), strlen("element face")) != 0);
			if (temp == info_lines_num)
				tmp_line = "0";
			else
				tmp_line = tmp_line.substr(13, tmp_line.length() - 1);
			this->m_totalFaces = atoi(tmp_line.c_str());
			cout << "element face: " << this->m_totalFaces << endl;

			// go to end_header
			do
			{
				getline(file, tmp_line);
			} while (strncmp("end_header", tmp_line.c_str(), strlen("end_header")) != 0);
			//----------------------


			// read vertices
			try
			{
				cout << "malloc " << m_totalConnectedPoints << endl;
				mp_vertexXYZ = (float*)malloc(m_totalConnectedPoints * sizeof(float) * 3);
				mp_vertexNorm = (float*)malloc(m_totalConnectedPoints * sizeof(float) * 3);
				mp_vertexRGB = (float*)malloc(m_totalConnectedPoints * sizeof(float) * 3);
			}
			catch (char*)
			{
				return -1;
			}
			i = 0;
			for (int iterator = 0; iterator < this->m_totalConnectedPoints; iterator++)//3488
			{
				getline(file, tmp_line);
				ss.clear();
				ss << tmp_line;
				ss >> mp_vertexXYZ[i];
				ss >> mp_vertexXYZ[i + 1];
				ss >> mp_vertexXYZ[i + 2];
				ss >> mp_vertexNorm[i];
				ss >> mp_vertexNorm[i + 1];
				ss >> mp_vertexNorm[i + 2];
				ss >> mp_vertexRGB[i];
				ss >> mp_vertexRGB[i + 1];
				ss >> mp_vertexRGB[i + 2];

				i += 3;
			}

			// read faces
			try
			{
				cout << "malloc " << m_totalFaces << endl;
				triangles = (float*)malloc(m_totalFaces * sizeof(float) * 9);
				triangle_norms = (float*)malloc(m_totalFaces * sizeof(float) * 9);
				colors = (float*)malloc(m_totalFaces * sizeof(float) * 12);
			}
			catch (char*)
			{
				return -1;
			}
			i = 0;
			for (int iterator = 0; iterator < this->m_totalFaces; iterator++)//6920
			{
				//fgets(buffer, 300, file);
				getline(file, tmp_line);
				ss.clear();
				ss << tmp_line;
				int tmp_val;
				ss >> tmp_val;

				if (tmp_val == 3)
				{
					int vertex1 = 0, vertex2 = 0, vertex3 = 0;
					ss >> vertex1;
					ss >> vertex2;
					ss >> vertex3;


					triangles[9*iterator+0] = mp_vertexXYZ[3 * vertex1 + 0];
					triangles[9*iterator+1] = mp_vertexXYZ[3 * vertex1 + 1];
					triangles[9*iterator+2] = mp_vertexXYZ[3 * vertex1 + 2];
					triangles[9*iterator+3] = mp_vertexXYZ[3 * vertex2 + 0];
					triangles[9*iterator+4] = mp_vertexXYZ[3 * vertex2 + 1];
					triangles[9*iterator+5] = mp_vertexXYZ[3 * vertex2 + 2];
					triangles[9*iterator+6] = mp_vertexXYZ[3 * vertex3 + 0];
					triangles[9*iterator+7] = mp_vertexXYZ[3 * vertex3 + 1];
					triangles[9*iterator+8] = mp_vertexXYZ[3 * vertex3 + 2];


					colors[12*iterator+0] = (mp_vertexRGB[3 * vertex1] / 255.0f);
					colors[12*iterator+1] = (mp_vertexRGB[3 * vertex1 + 1] / 255.0f);
					colors[12*iterator+2] = (mp_vertexRGB[3 * vertex1 + 2] / 255.0f);
					colors[12*iterator+3] = 1.f;
					colors[12*iterator+4] = (mp_vertexRGB[3 * vertex2] / 255.0f);
					colors[12*iterator+5] = (mp_vertexRGB[3 * vertex2 + 1] / 255.0f);
					colors[12*iterator+6] = (mp_vertexRGB[3 * vertex2 + 2] / 255.0f);
					colors[12*iterator+7] = 1.f;
					colors[12*iterator+8] = (mp_vertexRGB[3 * vertex3] / 255.0f);
					colors[12*iterator+9] = (mp_vertexRGB[3 * vertex3 + 1] / 255.0f);
					colors[12*iterator+10] = (mp_vertexRGB[3 * vertex3 + 2] / 255.0f);
					colors[12*iterator+11] = 1.f;

					// colors[12*iterator+0] = 1.f; // (mp_vertexRGB[3 * vertex1] / 255.0f);
					// colors[12*iterator+1] = 1.f; // (mp_vertexRGB[3 * vertex1 + 1] / 255.0f);
					// colors[12*iterator+2] = 1.f; // (mp_vertexRGB[3 * vertex1 + 2] / 255.0f);
					// colors[12*iterator+3] = 1.f; // 1.f;
					// colors[12*iterator+4] = 1.f; // (mp_vertexRGB[3 * vertex2] / 255.0f);
					// colors[12*iterator+5] = 1.f; // (mp_vertexRGB[3 * vertex2 + 1] / 255.0f);
					// colors[12*iterator+6] = 1.f; // (mp_vertexRGB[3 * vertex2 + 2] / 255.0f);
					// colors[12*iterator+7] = 1.f; // 1.f;
					// colors[12*iterator+8] = 1.f; // (mp_vertexRGB[3 * vertex3] / 255.0f);
					// colors[12*iterator+9] = 1.f; // (mp_vertexRGB[3 * vertex3 + 1] / 255.0f);
					// colors[12*iterator+10] = 1.f; // (mp_vertexRGB[3 * vertex3 + 2] / 255.0f);
					// colors[12*iterator+11] = 1.f; // 1.f;

					triangle_norms[9*iterator+0] = (mp_vertexNorm[3 * vertex1]);
					triangle_norms[9*iterator+1] = (mp_vertexNorm[3 * vertex1 + 1]);
					triangle_norms[9*iterator+2] = (mp_vertexNorm[3 * vertex1 + 2]);
					triangle_norms[9*iterator+3] = (mp_vertexNorm[3 * vertex2]);
					triangle_norms[9*iterator+4] = (mp_vertexNorm[3 * vertex2 + 1]);
					triangle_norms[9*iterator+5] = (mp_vertexNorm[3 * vertex2 + 2]);
					triangle_norms[9*iterator+6] = (mp_vertexNorm[3 * vertex3]);
					triangle_norms[9*iterator+7] = (mp_vertexNorm[3 * vertex3 + 1]);
					triangle_norms[9*iterator+8] = (mp_vertexNorm[3 * vertex3 + 2]);

				}


				i += 3;
			}

			file.close();
			cout << filename << " loaded OK!" << endl;

		}

		else
		{
			cout << "File can't be opened" << endl;
		}
	}
	else
	{
		cout << "File does not have a .PLY extension. " << endl;
	}

	return 0;
}

void ply_loader::update_mdbatch(GLBatch& batch)
{
	cout << m_totalFaces << endl;
	batch.Begin(GL_TRIANGLES, m_totalFaces * 3); // GL_TRIANGLES GL_TRIANGLE_STRIP GL_TRIANGLE_FAN
	batch.CopyVertexData3f(triangles);
	batch.CopyNormalDataf(triangle_norms);
	batch.CopyColorData4f(colors);
	batch.End();
}

//void ply_loader::Draw() //implemented in GLPainter, not called again
//{
//	cout << "Drawing ..." << endl;
//	if (m_ModelData.vecFaceTriangles.empty())
//	{
//		cout << m_ModelData.vecFaceTriangles.size() << endl;
//		cout << "model data is null" << endl;
//		while (true);
//		exit(-1);
//	}
//
//	glEnableClientState(GL_VERTEX_ARRAY);
//	glEnableClientState(GL_NORMAL_ARRAY);
//	glEnableClientState(GL_COLOR_ARRAY);
//	glVertexPointer(3, GL_FLOAT, 0, m_ModelData.vecFaceTriangles.data());
//	glColorPointer(3, GL_FLOAT, 0, m_ModelData.vecFaceTriangleColors.data());
//	glNormalPointer(GL_FLOAT, 0, m_ModelData.vecNormals.data());
//	glDrawArrays(GL_TRIANGLES, 0, m_ModelData.iTotalConnectedTriangles);
//	glDisableClientState(GL_VERTEX_ARRAY);
//	glDisableClientState(GL_NORMAL_ARRAY);
//	glDisableClientState(GL_COLOR_ARRAY);
//}