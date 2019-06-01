#include "xmlreader.h"

// read map from xml file, return a vector of Vertex
thrust::host_vector<Vertex> read_xml_map(const char* filename) {
	// result initial
	thrust::host_vector<Vertex> result;

	// open file
	TiXmlDocument doc(filename);
	if (!doc.LoadFile()) {
		cout << "Unable to open " << filename << "!" << endl;
		throw;
	}

	// read pointer init
	TiXmlHandle hDoc(&doc);
	TiXmlElement* pElem = hDoc.FirstChildElement().Element();
	TiXmlHandle hRoot(pElem);

	// for each vertex
	TiXmlElement* nodeElem = hRoot.FirstChild("graph").FirstChild("vertex").Element();
	for (int count = 0; nodeElem; nodeElem = nodeElem->NextSiblingElement(), count++) {
		// vertex init
		Vertex _v;
		_v.id = count;

		// read init
		TiXmlHandle subnode(nodeElem);
		TiXmlElement* subElem = subnode.FirstChild("edge").Element();

		// read each edge
		for (int subcount = 0; subElem; subElem = subElem->NextSiblingElement(), subcount++) {
			// read data
			float _dis;
			subElem->QueryFloatAttribute("cost", &_dis);
			int _id = atoi(subElem->GetText());

			// if skipped, add an empty one(self to self distance)
			while (_id != subcount) {
				subcount++;
				_v.distances.push_back(-1);
			}
			_v.distances.push_back(_dis);
		}

		// push into result
		result.push_back(_v);
	}

	return result;
}