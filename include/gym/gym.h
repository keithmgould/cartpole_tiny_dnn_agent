#ifndef __GYM_H__
#define __GYM_H__
// Caffe uses boost::shared_ptr (as opposed to std::shared_ptr), so do we.
#include "tiny_dnn/tiny_dnn.h"
#include <boost/shared_ptr.hpp>
#include <vector>

namespace Gym {

	struct State {
		// std::vector<float> observation; // get observation_space() to make sense of this data
		tiny_dnn::vec_t observation; // get observation_space() to make sense of this data
		float reward;
		bool done;
		std::string info;
	};

	struct Space {
		enum SpaceType {
			DISCRETE,
			BOX,
		} type;

		int weighted_sample(tiny_dnn::vec_t& prediction); // chooses an action based on the output probabilities
		std::vector<int>   box_shape; // Similar to Caffe blob shape, for example { 64, 96, 3 } for 96x64 rgb image.
		std::vector<float> box_high;
		std::vector<float> box_low;

		int discreet_n;
	};

	class Environment {
	public:
		virtual boost::shared_ptr<Space> action_space() =0;
		virtual boost::shared_ptr<Space> observation_space() =0;

		virtual void reset(State& save_initial_state_here) =0;

		virtual void step(const int action, bool render, State& save_state_here) =0;

		virtual void monitor_start(const std::string& directory, bool force, bool resume) =0;
		virtual void monitor_stop() =0;
	};

	class Client {
	public:
		virtual boost::shared_ptr<Environment> make(const std::string& name) =0;
	};

	extern boost::shared_ptr<Client> client_create(const std::string& addr, int port);

} // namespace

#endif // __GYM_H__
