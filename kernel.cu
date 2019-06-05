// cuda lib included in it
#include "xmlreader.h"

#include <vector>
// for random
#include <stdlib.h>
#include <time.h>
#define random(x) (rand()%x)
#define min(x,y) (x < y ? x : y)
#define max(x,y) (x > y ? x : y)

// make a random sequence that satisfy TSP.
// input a vector of vertex, return a vector contains the list.
//thrust::host_vector<int> make_random_sequence(thrust::host_vector<Vertex> list) {
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
		float _distance = maps[current].distances[next];
		if (_distance < 0) {
			printf("Warning: distance from %d to %d is not positive(%.5f), but it's on the sequence.\n",current, _distance, next);
			_distance = 0;
		}
		result += _distance;
	}
	return result;
}

// serial TSP solver
// input the map and some args, return a vector with the (maybe) best way.
thrust::host_vector<int> serial_TSP(thrust::host_vector<Vertex> maps, int max_trial = 5000, int max_retry = 1000) {
	const int origin_heat = 10000;
	const float deheat = 0.95;

	// initialize
	int heat = origin_heat;
	thrust::host_vector<int> sequence = make_random_sequence(maps);
	float seq_length = calculate_distance(maps, sequence);
	auto seq_size = sequence.size();
	int refuse_times = 0;

	// begin to run
	while (max_trial--) {
		// initial

		// pointer
		int first_point = 0;
		int next_point = 0;
		// edge's id
		int first_begin = 0;
		int first_end = 0;
		int next_begin = 0;
		int next_end = 0;
		// distances
		float distance_fb_to_fe = 0;
		float distance_nb_to_ne = 0;
		float distance_fb_to_nb = 0;
		float distance_fe_to_ne = 0;

		// find two point to exchange
		// from a[first_begin - first_end]bc[next_begin - next_end]d
		// to   a[first_begin - next_begin]cb[first_end - next_end]d
		while (true) {
			int _temp_1 = random(seq_size);
			int _temp_2 = random(seq_size);
			first_point = min(_temp_1, _temp_2);
			next_point = max(_temp_1, _temp_2);

			// judge whether changeable
			
			// too near
			if (next_point - first_point < 2) {
				continue;
			}

			// unable to connect
			first_begin = sequence[first_point];
			first_end = sequence[(first_point + 1) % seq_size];
			next_begin = sequence[next_point];
			next_end = sequence[(next_point + 1) % seq_size];
			distance_fb_to_nb = maps[first_begin].distances[next_begin];
			distance_fe_to_ne = maps[first_end].distances[next_end];
			if (distance_fb_to_nb >= 0 && distance_fe_to_ne >= 0) {
				break;
			}
		}
		
		// calculate origin distance
		distance_fb_to_fe = maps[first_begin].distances[first_end];
		distance_nb_to_ne = maps[next_begin].distances[next_end];

		// calculate delta
		float delta = distance_fb_to_nb + distance_fe_to_ne - distance_fb_to_fe - distance_nb_to_ne;
		// decide whether to accept it 
		// if delta >= 0, chance to accept it
		bool accept = (delta < 0);
		if (!accept) {
			int accept_chance = exp(-delta / heat) * 10000;
			accept = random(10000) < accept_chance;
		}

		if (accept) {
			refuse_times = 0;
			seq_length += delta;
			heat *= deheat;
			// make new seq
			sequence[(first_point + 1) % seq_size] = next_begin;
			sequence[(next_point) % seq_size] = first_end;
			// reverse [(first_point + 2) .. (next_point - 1)]
			int reverse_begin = first_point + 2;
			int reverse_end = next_point - 1;
			while (reverse_begin < reverse_end) {
				int _temp = sequence[reverse_begin];
				sequence[reverse_begin] = sequence[reverse_end];
				sequence[reverse_end] = _temp;
				reverse_begin++;
				reverse_end--;
			}
			printf("%d: Current length is %.5f\n", max_trial, seq_length);
		}
		else {
			refuse_times++;
			max_trial++;
			// too much trial
			if (refuse_times >= max_retry) {
				break;
			}
		}
	}

	return sequence;
}

// kernal part of TSP gpu-solver
// input:
//   map: a one-dimension vector on behalf of a 2d matrix
//        use map[y * map_width + x] to fetch it.
//   map_width: the width of the map
//              the map is a square, so width = height
//   max_trial: max trial times in this function
//   max_retry: max retry count for each trial
//              if reach the max_retry, it will stop in advanced
//   parallel_count: count of computation in a row
// output:
//   is_refused: a 1d vector recording each thread's return status.
//               if it's true, the thread stop in advanced.
// input/output: 
//   sequences_list: a 1d vector on behalf of a 2d matrix
//                   use sequences_list[tid * map_width] to fetch it (length map_width)
__global__ void gpu_TSP_kernel(
	float* map, const int map_width,
	int max_trial, int max_retry, int parallel_count,
	bool* is_refused, int* sequences_list
) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	// TODO
}

// host part of TSP gpu-solver
// input:
//   maps: a vector of vertex ordered by its id
//   max_trial: max trial times
//   max_retry: max retry count for each trial
//              if reach the max_retry, it will stop in advanced
//   parallel_count: count of computation in a row
//   trial_per_loop: trial times for per loop
// output: a vector with the (maybe) best way.
thrust::host_vector<int> gpu_TSP_host(thrust::host_vector<Vertex> maps, int max_trial = 5000, int max_retry = 1000, int parallel_count = 128, int trial_per_loop = 100) {
	// make sequence
	thrust::host_vector<int> sequence = make_random_sequence(maps);

	// transform maps into kernel form
	auto map_size = maps.size();
	thrust::host_vector<float> host_distance_map;
	for (int i = 0; i < map_size; ++i) {
		thrust::host_vector<float> _v = maps[i].distances;
		host_distance_map.insert(host_distance_map.end(), _v.begin(), _v.end());
	}
	thrust::device_vector<float> device_distance_map = host_distance_map;

	// run loop
	while (max_trial > 0) {
		// transform sequence into kernel form(for each thread)
		thrust::host_vector<int> host_sequence;
		for (int i = 0; i < parallel_count; ++i) {
			host_sequence.insert(host_sequence.end(), sequence.begin(), sequence.end());
		}
		thrust::device_vector<int> device_sequence = host_sequence;

		// initialize refuse vector
		thrust::device_vector<bool> is_refused(parallel_count, false);

		// initialize trial times for this times
		int trial_times = max_trial > trial_per_loop ? trial_per_loop : max_trial;
		max_trial -= trial_times;

		// run kernel
		gpu_TSP_kernel (
			thrust::raw_pointer_cast(&device_distance_map[0]), map_size,
			trial_times, max_retry, parallel_count,
			thrust::raw_pointer_cast(&is_refused[0]), thrust::raw_pointer_cast(&device_sequence[0]));

		// find the best sequence
		// TODO

		// judge whether all-stop
		// TODO
	}
	return sequence;
}

// main function
int main() {
	thrust::host_vector<Vertex> maps = read_xml_map("samples/xml/att532.xml");
	//thrust::host_vector<int> way_result = serial_TSP(maps, 100000, 10000);
	gpu_TSP_host(maps);

	return 0;
}