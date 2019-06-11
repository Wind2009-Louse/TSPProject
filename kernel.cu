// cuda lib included in it
#include "xmlreader.h"
#include "curand_kernel.h"

#include <vector>
#include <thrust/copy.h>

// for random
#include <stdlib.h>
#include <time.h>
#define random(x) (rand()%x)
#define curandom(gene, x) ((int)(curand_uniform(&gene) * x) % x)

// for max/min
#define min(x,y) (x < y ? x : y)
#define max(x,y) (x > y ? x : y)

// for time-count
#include <ctime>

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
thrust::host_vector<int> serial_TSP(thrust::host_vector<Vertex> maps, 
	int max_retry = 2, int max_refuse = 5, 
	float origin_heat = 50, float deheat = 0.95, int beta = 1) {
	// initialize
	float heat = origin_heat;
	thrust::host_vector<int> sequence = make_random_sequence(maps);
	float seq_length = calculate_distance(maps, sequence);
	auto seq_size = sequence.size();
	int max_trial_per_t = beta * seq_size * seq_size;

	int retry_times = 0;
	int refuse_times = 0;
	int trial_per_t = 0;
	bool changed_in_t = false;

	// begin to run
	while (heat > 0.0001) {
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
			changed_in_t = true;
			retry_times = 0;

			seq_length += delta;
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
		}
		if (!accept) {
			retry_times++;
		}
		trial_per_t++;
		// too much trial
		if (retry_times >= max_retry || trial_per_t >= max_trial_per_t) {
			trial_per_t = 0;
			if (!changed_in_t) printf("(Early-stop)");
			else printf("(Normally)");
			printf("%.3f: Current length is %.5f\n", heat, seq_length);
			heat *= deheat;
			if (!changed_in_t) {
				if (++refuse_times > max_refuse) {
					break;
				}
			} else{
				refuse_times = 0;
			}
			changed_in_t = false;
		}
	}

	return sequence;
}

// random initializer
// input a seed and count, then output a vector of initialized curandState(To avoid over-initial)
__global__ void random_initial(int seed, int parallel_count, curandState* gene_list) {
	// init
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// random init
	for (int i = tid; i < parallel_count; i += blockDim.x) {
		curand_init(seed, tid, 0, &gene_list[tid]);
	}
}

// kernal part of TSP gpu-solver
// input:
//   map: a one-dimension vector on behalf of a 2d matrix
//        use map[y * map_width + x] to fetch it.
//   map_width: the width of the map
//              the map is a square, so width = height
//   heat: heat of possibility
//   random_seed: seed for random function
//   parallel_count: count of computation in a row
// output:
//   is_refused: a 1d vector recording each thread's return status.
//               if it's true, the thread stop in advanced.
// input/output: 
//   sequences_list: a 1d vector on behalf of a 2d matrix
//                   use sequences_list[tid * map_width] to fetch it (length map_width)
__global__ void gpu_TSP_kernel(
	float* map, const int map_width,
	float heat, int parallel_count,
	curandState* rand_genes, bool* is_refused,
	int* sequences_list, float* sequences_length
) {
	// init
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// for each parallel times
	// if threads is not enough, it needs to run several times
	for (int pid = tid; pid < parallel_count; pid += blockDim.x) {
		// initial
		is_refused[pid] = false;
		int* sequence_head = new int[map_width];
		for (int i = 0; i < map_width; ++i) {
			sequence_head[i] = sequences_list[pid * map_width + i];
		}

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
			int _temp_1 = curandom(rand_genes[pid],map_width);
			int _temp_2 = curandom(rand_genes[pid], map_width);
			first_point = min(_temp_1, _temp_2);
			next_point = max(_temp_1, _temp_2);

			// judge whether changeable

			// too near
			if (next_point - first_point < 2) {
				continue;
			}

			// unable to connect
			first_begin = sequence_head[first_point];
			first_end = sequence_head[(first_point + 1) % map_width];
			next_begin = sequence_head[next_point];
			next_end = sequence_head[(next_point + 1) % map_width];
			distance_fb_to_nb = map[first_begin * map_width + next_begin];
			distance_fe_to_ne = map[first_end * map_width + next_end];
			if (distance_fb_to_nb >= 0 && distance_fe_to_ne >= 0) {
				break;
			}
		}

		// calculate origin distance
		distance_fb_to_fe = map[first_begin * map_width + first_end];
		distance_nb_to_ne = map[next_begin * map_width + next_end];

		// calculate delta
		float delta = distance_fb_to_nb + distance_fe_to_ne - distance_fb_to_fe - distance_nb_to_ne;
		// decide whether to accept it 
		// if delta >= 0, chance to accept it
		bool accept = (delta < 0);
		if (!accept) {
			int accept_chance = exp(-delta / heat) * 10000;
			accept = curandom(rand_genes[pid],10000) < accept_chance;
		}

		if (accept) {
			// init
			sequences_length[pid] += delta;

			// make new seq
			sequence_head[(first_point + 1) % map_width] = next_begin;
			sequence_head[(next_point) % map_width] = first_end;
			// reverse [(first_point + 2) .. (next_point - 1)]
			int reverse_begin = (first_point + 2) % map_width;
			int reverse_end = (next_point - 1) % map_width;
			while (reverse_begin < reverse_end) {
				int _temp = sequence_head[reverse_begin];
				sequence_head[reverse_begin] = sequence_head[reverse_end];
				sequence_head[reverse_end] = _temp;
				reverse_begin++;
				reverse_end--;
			}
		}
		else {
			is_refused[pid] = true;
		}
		
		for (int i = 0; i < map_width; ++i) {
			sequences_list[pid * map_width + i] = sequence_head[i];
		}
		delete sequence_head;
	}

}

// host part of TSP gpu-solver
// input:
//   maps: a vector of vertex ordered by its id
//   heat: heat of possibility
//   parallel_count: count of computation in a row
//   trial_per_loop: trial times for per loop
// output: a vector with the (maybe) best way.
thrust::host_vector<int> gpu_TSP_host(
	thrust::host_vector<Vertex> maps,
	float heat = 50, float deheat = 0.95, float beta = 1, 
	int parallel_count = 512
) {
	// transform maps into kernel form
	auto map_size = maps.size();
	thrust::host_vector<float> host_distance_map;
	for (int i = 0; i < map_size; ++i) {
		thrust::host_vector<float> _v = maps[i].distances;
		host_distance_map.insert(host_distance_map.end(), _v.begin(), _v.end());
	}
	thrust::device_vector<float> device_distance_map = host_distance_map;

	// initialize refuse vector
	thrust::device_vector<bool> device_is_refused(parallel_count, false);

	// make sequence
	thrust::host_vector<int> best_sequence;
	float best_length = -1;
	thrust::host_vector<int> host_sequences;
	thrust::host_vector<float> host_sequences_length;
	for (int i = 0; i < parallel_count; ++i) {
		thrust::host_vector<int> _sequence = make_random_sequence(maps);
		float _length = calculate_distance(maps, _sequence);
		host_sequences.insert(host_sequences.end(), _sequence.begin(), _sequence.end());
		host_sequences_length.push_back(_length);
	}
	// transform sequence into kernel form(for each thread)
	thrust::device_vector<int> device_sequence = host_sequences;
	thrust::device_vector<float> device_sequences_length = host_sequences_length;

	// random initialize
	thrust::device_vector<curandState> rand_genes(parallel_count);
	random_initial << <1, 512 >> > ((int)time(0), parallel_count, thrust::raw_pointer_cast(&rand_genes[0]));
	cudaDeviceSynchronize();

	// run loop
	int refused_times = 0;
	int trial_per_t = 0;
	int max_trial_per_t = beta * map_size * map_size;
	while (heat > 0.0001) {
		// first initialize

		// have changed in this T
		bool changed_in_t = true;
		while (true) {
			// run kernel
			gpu_TSP_kernel << <1, 512 >> > (
				thrust::raw_pointer_cast(&device_distance_map[0]), map_size,
				heat, parallel_count,
				thrust::raw_pointer_cast(&rand_genes[0]),
				thrust::raw_pointer_cast(&device_is_refused[0]),
				thrust::raw_pointer_cast(&device_sequence[0]),
				thrust::raw_pointer_cast(&device_sequences_length[0]));
			cudaDeviceSynchronize();

			// error check
			cudaError_t error = cudaGetLastError();
			const char* err_msg = cudaGetErrorString(error);
			if (strcmp("no error", err_msg) != 0) {
				printf("CUDA error: %s\n", err_msg);
			}

			// get result
			host_sequences.assign(device_sequence.begin(), device_sequence.end());
			host_sequences_length.assign(device_sequences_length.begin(), device_sequences_length.end());
			thrust::host_vector<bool> host_is_refused = device_is_refused;

			// judge whether all stop and find the best one
			bool has_accept = false;
			for (int i = 0; i < parallel_count; ++i) {
				if (!host_is_refused[i]) {
					has_accept = true;

					// judge
					float distance = host_sequences_length[i];
					if (best_length < 0 || distance < best_length) {
						// update sequence
						thrust::host_vector<int> _seq;
						_seq.insert(_seq.end(), host_sequences.begin() + i * map_size, host_sequences.begin() + (i + 1)*map_size);
						best_sequence.assign(_seq.begin(), _seq.end());
						best_length = distance;
					}
				}
			}
			trial_per_t += parallel_count;

			// if all_retry or too much trial
			if (!has_accept || trial_per_t >= max_trial_per_t) {
				// no change in this T
				if (!changed_in_t) {
					refused_times++;
				}
				if (!has_accept) printf("(Early-stop)");
				else printf("(Normally)");
				break;
			}
			else {
				refused_times = 0;
			}
			changed_in_t = true;
		}

		if (refused_times >= 10) {
			break;
		}

		// deheat
		printf("%.3f: Current best length is %.3f\n", heat, best_length);
		heat *= deheat;
		trial_per_t = 0;
	}
	return best_sequence;
}

// main function
int main() {
	clock_t ck, ck_2;
	//thrust::host_vector<Vertex> maps = read_xml_map("samples/xml/att532.xml");
	thrust::host_vector<Vertex> maps = read_xml_map("samples/xml/a280.xml");

	ck = clock();
	thrust::host_vector<int> serial_result = serial_TSP(maps);
	ck_2 = clock();
	printf("Serial time: %d\n", ck_2 - ck);
	printf("Serial length: %.5f\n", calculate_distance(maps, serial_result));

	ck = clock();
	thrust::host_vector<int> parallel_result = gpu_TSP_host(maps);
	ck_2 = clock();
	printf("Parallel time: %d\n", ck_2 - ck);
	printf("Parallel length: %.5f\n", calculate_distance(maps, parallel_result));

	printf("Run Successfully.\n");
	system("pause");

	return 0;
}