// speeds up compilation but no save/load of networks :/
#define CNN_NO_SERIALIZATION
#define GAMMA 0.99

#include "tiny_dnn/tiny_dnn.h"
#include "include/gym/gym.h"
#include <algorithm>
#include <cmath>
#include <numeric>
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

// take a state and run it through the network.
// return output probabilities
static vec_t forward_prop(Gym::State& state, network<sequential>& net){
	return net.predict(state.observation);
}

void apply_discount(std::vector<float>& rewards)
{
	std::vector<float> discounted;
	float R = 0;
	for(auto reward : rewards)
	{
		R = reward + GAMMA * R;
		discounted.insert(discounted.begin(), R);
	}

	rewards = discounted;
}

void set_mean_to_zero(std::vector<float>& rewards)
{
	double meany = 0;
	auto lambda = [&](double a, double b){return a + b / rewards.size(); };
	meany = std::accumulate(rewards.begin(), rewards.end(), 0.0, lambda);

	for(auto& reward : rewards)
		reward -= meany;
}

float compute_variance(std::vector<float>& rewards)
{
	float variance = 0;
	for(auto& reward : rewards)

		// assumes mean is zero, so reward - mean == reward.
		// can take a var eventually...
		variance += pow(reward, 2);

	return variance / rewards.size();
}

float compute_standard_deviation(float &variance)
{
	return sqrt(variance);
}

void set_std_dev_to_one(std::vector<float>& rewards)
{
	float variance = compute_variance(rewards);
	float standard_dev = compute_standard_deviation(variance);

	for(auto&reward : rewards)
		reward /= standard_dev;
}

/*
    Preprocessing does four things:
      0) reverse order of rewards (not sure needed since all rewards in this case are 1s)
			1) applies a discount. Related to credit assignment problem.
			2) set mean to zero. Related to reducing variance.
			3) set std dev to one. Related to reducing variance.

			Parts 2&3 are related to the 'baseline' of the REINFORCE algorithm.
*/
void preprocess_rewards(std::vector<float>& rewards)
{
	std::reverse(std::begin(rewards), std::end(rewards));
	apply_discount(rewards);
	set_mean_to_zero(rewards);
	set_std_dev_to_one(rewards);
}

/*
	Creates a one-hot vector for each action taken
*/
std::vector<vec_t> prepare_desired_out(std::vector<label_t>& actions)
{
	std::vector<vec_t> desired_out;
	for(typename std::vector<label_t>::const_iterator it = actions.begin(); it != actions.end(); ++it)
	{
		// std::cout << *it << ",";
		if((int) *it == 1){
			desired_out.push_back({1, 0});
		}else{
			desired_out.push_back({0, 1});
		}
	}
	// std::cout << std::endl;

	// for(std::vector<vec_t>::const_iterator it = desired_out.begin(); it != desired_out.end(); ++it)
	// {
	// 	std::cout << "(" << (*it)[0] << "," << (*it)[1] <<  ")";
	// }
	// std::cout << std::endl;

	return desired_out;
}

template<class T>
void print_vector(std::vector<T> &data, const std::string& var_name)
{
	std::cout << var_name << "--------------------" << std::endl;
	for(typename std::vector<T>::const_iterator it = data.begin(); it != data.end(); ++it)
		std::cout << *it << ',';
	std::cout << std::endl;
}

/*
	http://tiny-dnn.readthedocs.io/en/latest/how_tos/How-Tos.html#train-the-model

	arguments:
		net: this is the tiny-dnn network.
		observations: these are the states across the time-steps
		desired_out: these were the actions taken that produced the reward
		rewards: these are the normalized rewards
*/
void train(network<sequential>& net, std::vector<tiny_dnn::vec_t>& observations, std::vector<vec_t>& desired_out,std::vector<float>& rewards)
{

	adam optimizer;

	size_t batch_size = 1;
	size_t epochs = 1;
	net.fit<mse>(optimizer, observations, desired_out, batch_size, epochs);
}

// used only for debugging
float determine_average_total_rewards(std::vector<float>& total_rewards)
{
	if(total_rewards.size() == 0) { return 0; }

	int min_count = std::min((int)total_rewards.size(), 20);
	return accumulate( total_rewards.end() - min_count, total_rewards.end(), 0.0)/min_count;
}

static void run_single_environment(
	const boost::shared_ptr<Gym::Client>& client,
	const std::string& env_id,
	int episodes_to_run)
{
	boost::shared_ptr<Gym::Environment> env = client->make(env_id);
	boost::shared_ptr<Gym::Space> action_space = env->action_space();
	boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

	network<sequential> net;

	// We are taking in an observation of 4 floats:
	//     - x pos
	//     - x vel
	//     - pole angle
	//     - pole angular vel)
	// We output 2 probabilities
	//     -  move LEFT
	//     -  move RIGHT
	net << fully_connected_layer(4,128, false) // has_bias = false
			<< relu()
			<< fully_connected_layer(128,2, false) // has_bias = false
			<< softmax();

	std::vector<tiny_dnn::vec_t> observations; // each vector element holds 4 observations
	std::vector<float> rewards;
	std::vector<float> total_rewards; // sum of rewards per episode. Not the "return", just the basic sum.
	std::vector<tiny_dnn::label_t> actions; // chosen actions
	float avg_total_rewards = 0;

	for (int e=0; e<episodes_to_run; ++e) {
		Gym::State state;
		env->reset(state);
		float total_reward = 0;
		int total_steps = 0;
		std::vector<vec_t> desired_out;
		vec_t prediction;

		// clear our episode memory
		observations.clear();
		rewards.clear();
		actions.clear();

		while (1) {
			prediction = forward_prop(state, net);
			int action = action_space->weighted_sample(prediction);
			env->step(action, true, state); // state passed in so it can be updated
			// TODO: add penalty for deviating on X pos
			total_reward += state.reward;
			total_steps += 1;

			// store observations, actions and rewards
			actions.push_back((label_t) action);
			observations.push_back(state.observation);
			rewards.push_back(state.reward);

			if (state.done) break;
		}

		// normalize rewards
		preprocess_rewards(rewards);

		// create 1-hot vector (based on actions taken) to represent desired_out
		desired_out = prepare_desired_out(actions);

		train(net, observations, desired_out, rewards);

		// log some stuff for debugging...
		total_rewards.push_back(total_reward);
		avg_total_rewards = determine_average_total_rewards(total_rewards);
		printf("%s episode %i finished in %i steps with reward %0.2f and avg reward %0.2f\n",
			env_id.c_str(), e, total_steps, total_reward, avg_total_rewards);
	}
}

int main(int argc, char** argv)
{
	try {
		boost::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
		run_single_environment(client, "CartPole-v0", 1000);
	} catch (const std::exception& e) {
		fprintf(stderr, "ERROR: %s\n", e.what());
		return 1;
	}

	return 0;
}
