from collections import namedtuple


class Transition(namedtuple):
    def __init__(self, current_state, action, reward, next_state):
        """
        Args: 
            current_state: the state of the judgement that the current action
                are made upon;
            action: the action has been take at current time step;
            reward: the reward receieved from the environment;
            next_state: the state returned after taking the action.
        """

        super(Transition,
              self).__init__("Transition",
                             ("state", "action", "reward", "next_state"))

        self["state"] = current_state
        self["action"] = action
        self["reward"] = reward
        self["next_state"] = next_state
