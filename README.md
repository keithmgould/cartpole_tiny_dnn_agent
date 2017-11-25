The goal here is to use the [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) library for reinforcement learning. The environment is handled by [OpenAI Gym](https://github.com/openai/gym), and the interface is through the [OpenAI Gym Http Server](https://github.com/openai/gym-http-api). The agent communicates with the environment using the c++ [OpenAI Gym Http client](https://github.com/openai/gym-http-api/tree/master/binding-cpp).


### Status:

The agent currently learns and performs much better than a random agent, but caps out at only a decent level of learning, and I'm not sure why.


### Dependencies:

0. Python (tested on 3.6.3)
0. OpenAI's gym
0. a c++ compiler
0. boost (needed by the gym http c++ client)

### Origin of the http client code

The code for the gym http client in this repo is (somewhat heavily) modified from [here](https://github.com/openai/gym-http-api/tree/master/binding-cpp).

### Inspiration:

I treated the code found in PyTorch's [example for REINFORCE](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py) as a template for the agent in this repo.

### Setup:

0. Ensure you have python on your machine, able to run the OpenAI http server. Directions [HERE](https://github.com/openai/gym-http-api#getting-started).
0. you can use URLs in your browser to test that the gym server is running
0. **Modify the Makefile** to suit your system and compiler. You will need to update the paths.
0. If you can run make, and the compiled 'agent' is created, you are golden.

### Running:

Once you run the agent, you *should* see the gym render the simulation. This is at least the case for running on a mac.

### Pseudocode:

The part I'm *least* confident about is the `desired_outs` section. This corresponds to [this](https://github.com/keithmgould/cartpole_tiny_dnn_agent/blob/master/agent.cpp#L89) method.

```
  Forever do
    state = env.reset
    states = actsions = rewards = []
    while in_episode?
      possible_actions = net.predict(state) //softmax (probabilities sum to 1)
      action = weighted_random_choice(possible_actions) (action is 0 or 1 for left or right)
      state, reward, in_episode? = env.step(action) // reward is always 1
      actions.push(action); state.push(state); reward.push(reward) // store things
    rewards = normalize(rewards) // mean=0, std=1
    desired_outs = one_hot_using_rewards(actions,rewards) // for ex: 1 => {1.3, 0}. 0 => {0, 0.7}
    for(int i = 0; i < actions.size; i++)
      net.train(state[i], desired_outs[i])
```
