#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include "xmlreader.h"

#include <iostream>
using namespace std;

int main() {
	thrust::host_vector<Vertex> maps = read_xml_map("a280.xml");

	return 0;
}