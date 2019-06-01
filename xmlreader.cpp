#include "xmlreader.h"

thrust::host_vector<Vertex> read_xml_map(const char* filename) {
	// initial
	thrust::host_vector<Vertex> result;

	// open file
	TiXmlDocument doc(filename);
	if (!doc.LoadFile()) {
		cout << "Unable to open " << filename << "!" << endl;
		throw;
	}

	// read
	TiXmlHandle hDoc(&doc);
	TiXmlElement* pElem = hDoc.FirstChildElement().Element();
	TiXmlHandle hRoot(pElem);

	TiXmlElement* nodeElem = hRoot.FirstChild("graph").FirstChild("vertex").Element();
	for (int count = 0; nodeElem; nodeElem = nodeElem->NextSiblingElement(), count++) {
		// init
		Vertex _v;
		_v.id = count;

		// read init
		TiXmlHandle subnode(nodeElem);
		TiXmlElement* subElem = subnode.FirstChild("edge").Element();

		// read each data
		for (int subcount = 0; subElem; subElem = subElem->NextSiblingElement(), subcount++) {
			float _dis;
			subElem->QueryFloatAttribute("cost", &_dis);
			int _id = atoi(subElem->GetText());
			if (_id != subcount) {
				subcount++;
				_v.distances.push_back(0);
			}
			_v.distances.push_back(_dis);
		}

		result.push_back(_v);
	}

	return result;
}