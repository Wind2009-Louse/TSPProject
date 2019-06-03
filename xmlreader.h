/*
Transform XML data into c++ data.

data source: https://wwwproxy.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/XML-TSPLIB/instances/
*/

#ifndef XMLREADER_H
#define XMLREADER_H

// cuda lib
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

// normal lib
#include <iostream>
#include "tinyxml/tinystr.h"
#include "tinyxml/tinyxml.h"
using namespace std;


class Vertex {
public:
	// current vertex's id
	int id;

	// a distance matrix, including the distance to other vertexes
	// specially, distances[id] < 0 means no way to vertex with such id
	thrust::host_vector<float> distances;
};

// read map from xml file, return a vector of Vertex
thrust::host_vector<Vertex> read_xml_map(const char* filename);

#endif