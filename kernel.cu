// cuda lib included in it
#include "xmlreader.h"
#include "curand_kernel.h"

#include <vector>
#include <thrust/copy.h>

// for random
#include <stdlib.h>
#include <time.h>
#define random(x) (rand()%(x))
#define curandom(gene, x) ((int)(curand_uniform(&gene) * (x)) % (x))

// for max/min
#define min(x,y) (x < y ? x : y)
#define max(x,y) (x > y ? x : y)

// for time-count
#include <ctime>

// some constant, used to compared between serial and cuda
const int MAX_RETRY = 100;
const int MAX_REFUSED = 5;
const float ORIGIN_HEAT = 100;
const float DEHEAT_PER = 0.95;
const float MIN_HEAT = 0.0001;
const int BETA = 3;

// middle DEBUG
#define OUTPUT_DEBUG

// make a random sequence that satisfy TSP.
// input a vector of vertex, return a vector contains the list.
//thrust::host_vector<int> make_random_sequence(thrust::host_vector<Vertex> list) {
thrust::host_vector<int> make_random_sequence(thrust::host_vector<Vertex> list) {
	// initialize
	thrust::host_vector<int> result;
	vector<bool> reached(list.size(), false);
	// use as <current_begining, current_ptr>
	vector<pair<int, int> > steps;
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
	// for each vertex
	for (int i = 0; i < s_size; ++i) {
		int current = sequence[i];
		int next = sequence[(i + 1) % s_size];
		// find the distance
		float _distance = maps[current].distances[next];
		if (_distance < 0) {
			printf("Warning: distance from %d to %d is not positive(%.5f), but it's on the sequence.\n",current, next, _distance);
			_distance = 0;
		}
		result += _distance;
	}
	return result;
}

/*
serial TSP solver
input:
* maps: a vector of vertex ordered by its id
* max_retry: max retry times in a T
* max_refused: max refused for a series of T
* origin_heat: heat of possibility
* deheat: deheat perc
* beta: used to calculate max run times per T
output: a vector with the (maybe) best way.
*/
thrust::host_vector<int> serial_TSP(thrust::host_vector<Vertex> maps, 
	int max_retry = MAX_RETRY, int max_refuse = MAX_REFUSED,
	float origin_heat = ORIGIN_HEAT, float deheat = DEHEAT_PER, int beta = BETA) {
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
	while (heat > MIN_HEAT) {
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
			int _temp_2 = (random(seq_size - 3) + _temp_1 + 2) % seq_size;
			first_point = min(_temp_1, _temp_2);
			next_point = max(_temp_1, _temp_2);

			// judge whether changeable

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
		// if not accepted, add retry
		if (!accept) {
			retry_times++;
		}
		trial_per_t++;

		// too much trial or too much retry
		if (retry_times >= max_retry || trial_per_t >= max_trial_per_t) {
			trial_per_t = 0;
			if (!changed_in_t) printf("(Early-stop)");
			else printf("(Normally)");
			printf("%.3f: Current length is %.5f\n", heat, seq_length);
			heat *= deheat;

			// if is refused
			if (!changed_in_t) {
				// if refused too much times
				if (++refuse_times > max_refuse) {
					break;
				}
			} else{
				// clear counter
				refuse_times = 0;
			}
			changed_in_t = false;
			retry_times = 0;
		}
	}

	// return result
	return sequence;
}

// random initializer
// input a seed and count, then output a vector of initialized curandState(To avoid over-initial)
__global__ void random_initial(int seed, curandState* gene_list) {
	// init
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// random init
	for (int i = tid; i < blockDim.x; i += blockDim.x) {
		curand_init(seed, tid, 0, &gene_list[tid]);
	}
}

/*
kernal part of tsp gpu-solver
input:
* map: a one-dimension vector on behalf of a 2d matrix
** use map[y * map_width + x] to fetch it.
* map_width: the width of the map
** the map is a square, so width = height
* max_retry: max retry times for each thread
** if exceed, it will be reset as the best one
* heat: heat of possibility
* beta: args for max_trial_times
* rand_genes: generators for random function
output:
* is_refused: a 1d vector recording each thread's return status.
** if it's true, the thread stop in advanced.
* tag: tag of running result: 0(normal), 1(early-stop), 2(refused)
input/output: 
* sequences_list: a 1d vector on behalf of a 2d matrix
** use sequences_list[tid * map_width] to fetch it (length map_width)
* sequences_length: lengths of each list
*/
__global__ void gpu_TSP_kernel(
	float* map, const int map_width, int max_retry, float heat, int beta,
	curandState* rand_genes, int* sequences_list, float* sequences_length,
	float* best_dis, int* best_seq,
	int* tag
) {
	// init
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int sequences_offset = tid * map_width;
	int max_trial_times = beta * map_width * map_width / blockDim.x;
	int retry_count = 0;
	bool make_better = false;

	// early-stop init
	__shared__ bool exist_running[1024];
	__shared__ bool better_check[1024];

	while (max_trial_times--) {
		// skip if refused
		if (retry_count >= max_retry) {
		}
		else {
			// find two point to exchange
			// from a[first_begin - first_end]bc[next_begin - next_end]d
			// to   a[first_begin - next_begin]cb[first_end - next_end]d
			int _temp_1 = curandom(rand_genes[tid], map_width);
			int _temp_2 = (_temp_1 + 2 + curandom(rand_genes[tid], map_width - 3)) % map_width;

			// pointer
			int first_point = min(_temp_1, _temp_2);
			int next_point = max(_temp_1, _temp_2);

			// edge's id
			int first_begin = sequences_list[sequences_offset + first_point];
			int first_end = sequences_list[sequences_offset + (first_point + 1) % map_width];
			int next_begin = sequences_list[sequences_offset + next_point];
			int next_end = sequences_list[sequences_offset + (next_point + 1) % map_width];

			// distances
			float distance_fb_to_nb = map[first_begin * map_width + next_begin];
			float distance_fe_to_ne = map[first_end * map_width + next_end];

			// judge whether changeable
			if (distance_fb_to_nb < 0 || distance_fe_to_ne < 0) {
				retry_count += 1;
			}
			else {
				// calculate origin distance
				float distance_fb_to_fe = map[first_begin * map_width + first_end];
				float distance_nb_to_ne = map[next_begin * map_width + next_end];

				// calculate delta
				float delta = distance_fb_to_nb + distance_fe_to_ne - distance_fb_to_fe - distance_nb_to_ne;
				// decide whether to accept it 
				// if delta >= 0, chance to accept it
				bool accept = (delta < 0);
				if (!accept) {
					int accept_chance = exp(-delta / heat) * 10000;
					accept = curandom(rand_genes[tid], 10000) < accept_chance;
				}

				if (accept) {
					// init
					sequences_length[tid] += delta;
					retry_count = 0;
					make_better = true;

					// make new seq
					sequences_list[sequences_offset + (first_point + 1) % map_width] = next_begin;
					sequences_list[sequences_offset + (next_point) % map_width] = first_end;
					// reverse [(first_point + 2) .. (next_point - 1)]
					int reverse_begin = (first_point + 2) % map_width;
					int reverse_end = (next_point - 1) % map_width;
					while (reverse_begin < reverse_end) {
						int _temp = sequences_list[sequences_offset + reverse_begin];
						sequences_list[sequences_offset + reverse_begin] = sequences_list[sequences_offset + reverse_end];
						sequences_list[sequences_offset + reverse_end] = _temp;
						reverse_begin++;
						reverse_end--;
					}
				}
				else {
					retry_count += 1;
				}
			}
		}

		// early-stop judge
		exist_running[tid] = retry_count < max_retry;
		better_check[tid] = make_better;
		__syncthreads();

		for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
			if (tid + i < blockDim.x) {
				exist_running[tid] = exist_running[tid] | exist_running[tid + i];
				better_check[tid] = better_check[tid] | better_check[tid + i];
			}
			__syncthreads();
		}
		if (!exist_running[0]) {
			break;
		}
	}

	// find the best
	__shared__ float _best_length[1024];
	__shared__ float _best_id[1024];
	_best_length[tid] = sequences_length[tid];
	_best_id[tid] = tid;
	__syncthreads();
	for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
		if (tid + i < blockDim.x) {
			float current_best = _best_length[tid];
			float next_length = _best_length[tid + i];
			_best_id[tid] = (next_length < current_best) ? _best_id[tid + i] : _best_id[tid];
			_best_length[tid] = (next_length < current_best) ? next_length : current_best;
		}
		__syncthreads();
	}

	// update the best
	__shared__ bool need_update;
	if (tid == 0) {
		need_update = false;
		if (_best_length[0] < best_dis[0]) {
			best_dis[0] = _best_length[0];
			need_update = true;
		}
	}
	__syncthreads();
	if (need_update) {
		int best_offset = _best_id[0] * map_width;
		for (int i = tid; i < map_width; i += blockDim.x) {
			best_seq[i] = sequences_list[best_offset + i];
		}
	}

	// reused the best
	if (retry_count >= max_retry) {
		for (int i = 0; i < map_width; ++i) {
			sequences_list[sequences_offset + i] = best_seq[i];
		}
	}

	// make tag by thread 0
	if (tid == 0) {
		if (!better_check[0]) {
			tag[0] = 2;
		}
		else if (!exist_running[0]) {
			tag[0] = 1;
		}
		else {
			tag[0] = 0;
		}
	}
}

/*
// host part of TSP gpu-solver
input:
* maps: a vector of vertex ordered by its id
* max_retry: max retry times in a T
* max_refused: max refused for a series of T
* heat: heat of possibility
* deheat: deheat perc
* beta: used to calculate max run times per T
* parallel_count: count of computation in a row
output: a vector with the (maybe) best way.
*/
thrust::host_vector<int> gpu_TSP_host(
	thrust::host_vector<Vertex> maps,
	int max_retry = MAX_RETRY, int max_refused = MAX_REFUSED,
	float heat = ORIGIN_HEAT, float deheat = DEHEAT_PER, float beta = BETA,
	int parallel_count = 1024
) {
	// transform maps into kernel form
	auto seq_size = maps.size();
	thrust::host_vector<float> host_distance_map;
	for (int i = 0; i < seq_size; ++i) {
		host_distance_map.insert(host_distance_map.end(), maps[i].distances.begin(), maps[i].distances.end());
	}
	thrust::device_vector<float> device_distance_map = host_distance_map;
	device_distance_map.resize(host_distance_map.size());

	// make sequence
	thrust::host_vector<int> host_sequences;
	thrust::host_vector<float> host_sequences_length;
	for (int i = 0; i < parallel_count; ++i) {
		thrust::host_vector<int> _sequence = make_random_sequence(maps);
		float _length = calculate_distance(maps, _sequence);
		host_sequences.insert(host_sequences.end(), _sequence.begin(), _sequence.end());
		host_sequences_length.push_back(_length);
	}

	// best record init
	thrust::host_vector<int> best_sequence(host_sequences.begin(), host_sequences.begin() + seq_size);
	thrust::device_vector<int> device_best_sequence = best_sequence;
	thrust::device_vector<float> device_best_length(1, host_sequences_length[0]);

	// transform sequence into kernel form(for each thread)
	thrust::device_vector<int> device_sequence = host_sequences;
	thrust::device_vector<float> device_sequences_length = host_sequences_length;

	// random initialize
	thrust::device_vector<curandState> rand_genes(parallel_count);
	random_initial << <1, parallel_count >> > ((int)time(0), thrust::raw_pointer_cast(&rand_genes[0]));

	// run loop
	int refused_times = 0;
	thrust::device_vector<int> result_tag(1,0);

	// ptr init
	float* map_ptr = thrust::raw_pointer_cast(&device_distance_map[0]);
	curandState* curand_ptr = thrust::raw_pointer_cast(&rand_genes[0]);
	int* seq_ptr = thrust::raw_pointer_cast(&device_sequence[0]);
	float* seq_length_ptr = thrust::raw_pointer_cast(&device_sequences_length[0]);
	float* best_seqlength_ptr = thrust::raw_pointer_cast(&device_best_length[0]);
	int* best_seq_ptr = thrust::raw_pointer_cast(&device_best_sequence[0]);
	int* tag_ptr = thrust::raw_pointer_cast(&result_tag[0]);

	clock_t ck, ck_2;
	ck = clock();
	while (heat > MIN_HEAT) {
		// run kernel
		gpu_TSP_kernel<<<1,parallel_count>>>(
			map_ptr, seq_size, max_retry, heat, beta, 
			curand_ptr, seq_ptr, seq_length_ptr, 
			best_seqlength_ptr, best_seq_ptr,
			tag_ptr);

		// error check
		cudaError_t error = cudaGetLastError();
		const char* err_msg = cudaGetErrorString(error);
		if (strcmp("no error", err_msg) != 0) {
			printf("CUDA error: %s\n", err_msg);
		}

		int tag = result_tag[0];

		best_sequence.assign(device_best_sequence.begin(), device_best_sequence.end());
		// no change in this T
		if (tag == 2) {
			refused_times++;
		}
#ifdef OUTPUT_DEBUG
		if (tag == 2) printf("(Refused)");
		else if (tag == 1) printf("(Early-stop)");
		else if (tag == 0) printf("(Normally)");
		else printf("(Unknown tag: %d)");
		float real_length = calculate_distance(maps, best_sequence);
		printf("%.3f: Current best length is %.3f\n", heat, real_length);
#endif // OUTPUT_DEBUG

		if (refused_times >= max_refused) {
			break;
		}

		// deheat
		heat *= deheat;
	}
	ck_2 = clock();
	printf("Time used in loop: %d\n", ck_2 - ck);
	
	return best_sequence;
}

// main function
int main() {
	// init
	srand((int)time(0));
	clock_t ck, ck_2;
	//thrust::host_vector<Vertex> maps = read_xml_map("samples/xml/att48.xml");
	//thrust::host_vector<Vertex> maps = read_xml_map("samples/xml/brg180.xml");
	//thrust::host_vector<Vertex> maps = read_xml_map("samples/xml/a280.xml");
	//thrust::host_vector<Vertex> maps = read_xml_map("samples/xml/att532.xml");
	thrust::host_vector<Vertex> maps = read_xml_map("samples/xml/d657.xml");

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