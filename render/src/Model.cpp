#include "Model.h"

#include <cstdio>

#include <iostream>
#include <fstream>
#include <string>

Model::Model() : m_vtx{std::make_shared<std::vector<glm::vec3>>()},
                 m_tri{std::make_shared<std::vector<glm::u32vec3>>()},
                 m_nrm{std::make_shared<std::vector<glm::vec3>>()} {}

// 仅支持v、f
bool Model::initModel(const std::filesystem::path& model_path, double scale, glm::vec3 shift, bool swap_xyz) {
	std::vector<glm::u32vec3> triset;
	std::vector<glm::vec3> rgbset;
	std::vector<glm::vec3> vtxset;
	std::vector<glm::vec2> texset;
	std::vector<glm::u32vec3> ttriset;

	int triIdx = 0;

	std::string string_path = model_path.string();
	const char* filePathCStr = string_path.c_str();

	FILE* fp = fopen(filePathCStr, "rt");
	if (fp == NULL) return false;

	char buf[1024];
	while (fgets(buf, 1024, fp)) {
		if (buf[0] == 'v' && buf[1] == ' ') {
			double x, y, z;
			double r, g, b;
			sscanf(buf + 2, "%lf%lf%lf%lf%lf%lf", &x, &y, &z, &r, &g, &b);
			if (swap_xyz)
			{
				vtxset.push_back(glm::vec3(z, x, y) * glm::vec3(scale,scale,scale) + shift);
				rgbset.emplace_back(glm::vec3(b, r, g));
			}
			else {
				vtxset.push_back(glm::vec3(x, y, z) * glm::vec3(scale, scale, scale) + shift);
				rgbset.emplace_back(glm::vec3(r, g, b));
			}
		}
		else

			if (buf[0] == 'v' && buf[1] == 't') {
				double x, y;
				sscanf(buf + 3, "%lf%lf", &x, &y);

				texset.push_back(glm::vec2(x, y));
			}
			else
				if (buf[0] == 'f' && buf[1] == ' ') {
					int id0, id1, id2, id3 = 0;
					int tid0, tid1, tid2, tid3 = 0;
					bool quad = false;

					int count = sscanf(buf + 2, "%d/%d", &id0, &tid0);
					char* nxt = strchr(buf + 2, ' ');
					sscanf(nxt + 1, "%d/%d", &id1, &tid1);
					nxt = strchr(nxt + 1, ' ');
					sscanf(nxt + 1, "%d/%d", &id2, &tid2);

					nxt = strchr(nxt + 1, ' ');
					if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9') {// quad
						if (sscanf(nxt + 1, "%d/%d", &id3, &tid3))
							quad = true;
					}

					id0--, id1--, id2--, id3--;
					tid0--, tid1--, tid2--, tid3--;

					triset.push_back(glm::u32vec3(id0, id1, id2));
					if (count == 2) {
						ttriset.push_back(glm::u32vec3(tid0, tid1, tid2));
					}

					if (quad) {
						triset.push_back(glm::u32vec3(id0, id2, id3));
						if (count == 2)
							ttriset.push_back(glm::u32vec3(tid0, tid2, tid3));
					}
					triIdx++;
				}
	}
	fclose(fp);

	if (triset.size() == 0 || vtxset.size() == 0)
		return false;
	for (int i = 0; i < vtxset.size(); i++) 
	{
		m_vtx->push_back(vtxset[i]);
	}
	for (int i = 0; i < triset.size(); i++) 
	{
		m_tri->push_back(triset[i]);
	}

	m_num_vtx = m_vtx->size();
	m_num_tri = m_tri->size();
	m_nrm->resize(m_num_vtx, glm::vec3{});

	uint32_t  vid1{ 0 };
	uint32_t  vid2{ 0 };
	uint32_t  vid3{ 0 };
	glm::vec3 vertex1{};
	glm::vec3 vertex2{};
	glm::vec3 vertex3{};
	glm::vec3 face_normal{};

	for (uint32_t i = 0; i < m_num_tri; i++) {
		vid1 = (*m_tri)[i].x;
		vid2 = (*m_tri)[i].y;
		vid3 = (*m_tri)[i].z;
		vertex1 = (*m_vtx)[vid1];
		vertex2 = (*m_vtx)[vid2];
		vertex3 = (*m_vtx)[vid3];

		face_normal = normalize(cross(vertex2 - vertex1, vertex3 - vertex1));

		(*m_nrm)[vid1] += face_normal;
		(*m_nrm)[vid2] += face_normal;
		(*m_nrm)[vid3] += face_normal;
	}

	for (uint32_t i = 0; i < m_num_vtx; i++) {
		(*m_nrm)[i] = normalize((*m_nrm)[i]);
	}
	return true;
}

void Model::memFree() {
    m_vtx.reset();
    m_tri.reset();
    m_nrm.reset();
}
