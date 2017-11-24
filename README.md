The goal here is to use the [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) library for reinforcement learning. The environment is handled by [OpenAI Gym](https://github.com/openai/gym), and the interface is through the [OpenAI Gym Http Server](https://github.com/openai/gym-http-api). The agent communicates with the environment using the c++ [OpenAI Gym Http client](https://github.com/openai/gym-http-api/tree/master/binding-cpp).


### Status:

The agent currently learns and performs much better than a random agent, but caps out at only a decent level of learning, and I'm not sure why.


### Dependencies:

0. Python (tested on 3.6.3)
0. OpenAI's gym
0. a c++ compiler
0. boost (needed by the gym http c++ client)

### Setup:

0. Ensure you have python on your machine, able to run the OpenAI http server. Directions [HERE](https://github.com/openai/gym-http-api#getting-started).
0. you can use URLs in your browser to test that the gym server is running
0. **Modify the Makefile** to suit your system and compiler. You will need to update the paths.
0. If you can run make, and the compiled 'agent' is created, you are golden.

### Running:

Once you run the agent, you *should* see the gym render the simulation. This is at least the case for running on a mac.
