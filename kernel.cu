// cuda lib included in it
#include "xmlreader.h"

// for random
#include <stdlib.h>
#include <time.h>
#define random(x) (rand()%x)

// make a random sequence that satisfy TSP.
// input a vector of vertex, return a vector contains the list.
thrust::host_vector<int> make_random_sequence(thrust::host_vector<Vertex> list) {
	// initialize
	thrust::host_vector<int> result;
	vector<bool> reached(list.size(), false);
	// use as <current_begining, current_ptr>
	vector<pair<int, int> > steps;
	srand((int)time(0));
	auto max_size = list.size();

	// since it's a ring, set the beginning as 0
	result.push_back(list[0].id);
	reached[0] = true;

	// next step's begining
	int next_beginning = random(max_size);
	steps.push_back(pair<int, int>(next_beginning, next_beginning));

	// dfs
	while(true){
		// break check
		auto result_size = result.size();
		if (result_size >= max_size || result_size == 0) {
			break;
		}

		pair<int, int> this_step = steps[steps.size() - 1];
		int current = result[result.size() - 1];
		int find_begining = this_step.first;
		int ptr = this_step.second;
		// find next position
		while (true) {
				// reached
			if (reached[ptr] 
				// unable to reach
				|| (list[current].distances[ptr] < 0)
				// unable to return to 0
				|| (max_size - result_size == 1 && list[ptr].distances[0] < 0)) {

				ptr = (ptr + 1) % max_size;
				// already search a loop
				if (ptr == find_begining) {
					// go back to last step
					reached[current] = false;
					result.pop_back();
					steps.pop_back();
					break;
				}
			}
			// find
			else {
				// fix current step
				steps[steps.size() - 1].second = (ptr + 1) % max_size;

				// next step
				reached[ptr] = true;
				result.push_back(ptr);
				int next_beginning = random(max_size);
				steps.push_back(pair<int, int>(next_beginning, next_beginning));
				break;
			}
		}
	}

	return result;
}

// calculate the total distance for a sequence.
// input a vector of vertex and the sequences, return the total length of sequence.
float calculate_distance(thrust::host_vector<Vertex> maps, thrust::host_vector<int> sequence) {
	float result = 0;
	auto s_size = sequence.size();
	for (int i = 0; i < s_size; ++i) {
		int current = sequence[i];
		int next = sequence[(i + 1) % s_size];
		float _distance = maps[i].distances[i + 1];
		if (_distance <= 0) {
			printf("Warning: distance from %d to %d is not positive, but it's on the sequence.\n",current, next);
			_distance = 0;
		}
		result += _distance;
	}
	return result;
}

// main function
int main() {
	thrust::host_vector<Vertex> maps = read_xml_map("a280.xml");
	thrust::host_vector<int> random_ways = make_random_sequence(maps);

	return 0;
}