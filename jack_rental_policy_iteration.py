from typing import Any
import pandas as pd
from scipy.stats import poisson

THETA: float = 1e-8
GAMMA: float = 0.9
ACTIONS: list[str] = ['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5']
POISSON_LAMBDA_RENTAL_CARS_1: float = 3.0
POISSON_LAMBDA_RETURN_CARS_1: float = 3.0
POISSON_LAMBDA_RENTAL_CARS_2: float = 4.0
POISSON_LAMBDA_RETURN_CARS_2: float = 2.0
EARNING_PER_CAR: float = 10.0
STATE_SIZE: int = 20
ACTION_COST: float = 2.0


class State:

    def __init__(self,
                 name: str,
                 state_1: int,
                 state_2: int
                 ):
        self.name = name
        self.state_1 = state_1
        self.state_2 = state_2
        self.V = 0.0
        self.PI = self._init_PI()
        self.R = self._calculate_R()
    
    def _init_PI(self):
        """+5 is location 1 recieves 5 cars, -5 is location 1 send 5 cars"""
        PI = {}
        for action in ACTIONS:
            action = int(action.strip())
            if action > 0:
                if ((self.state_2 >= action) and (self.state_1 + action <= STATE_SIZE)):
                    PI[str(action)] = 0.0
            elif action < 0:
                if ((self.state_1 >= -action) and (self.state_2 + -action <= STATE_SIZE)):
                    PI[str(action)] = 0.0
            elif action == 0:
                PI[str(action)] = 0.0
        for action in PI.keys():
            PI[action] = 1.0 / len(PI.keys())
        return PI
    
    def _calculate_R(self):
        R_1 = 0.0
        R_2 = 0.0
        rental_cars_1 = range(0, 40)
        arrive_cars_1 = range(0, 40)
        rental_cars_2 = range(0, 40)
        arrive_cars_2 = range(0, 40)
        for rental_car_1 in rental_cars_1:
            for arrive_car_1 in arrive_cars_1:
                net_change_1 = self.state_1 - rental_car_1 + arrive_car_1
                if net_change_1 >= 0:
                    R_1 += EARNING_PER_CAR * rental_car_1 * poisson.pmf(rental_car_1, POISSON_LAMBDA_RENTAL_CARS_1) * poisson.pmf(arrive_car_1, POISSON_LAMBDA_RETURN_CARS_1)
                elif net_change_1 < 0:
                    R_1 += EARNING_PER_CAR * (net_change_1 + rental_car_1) * poisson.pmf(rental_car_1, POISSON_LAMBDA_RENTAL_CARS_1) * poisson.pmf(arrive_car_1, POISSON_LAMBDA_RETURN_CARS_1)

        for rental_car_2 in rental_cars_2:
            for arrive_car_2 in arrive_cars_2:
                net_change_2 = self.state_2 - rental_car_2 + arrive_car_2
                if net_change_2 >= 0:
                    R_2 += EARNING_PER_CAR * rental_car_2 * poisson.pmf(rental_car_2, POISSON_LAMBDA_RENTAL_CARS_2) * poisson.pmf(arrive_car_2, POISSON_LAMBDA_RETURN_CARS_2)
                elif net_change_2 < 0:
                    R_2 += EARNING_PER_CAR * (net_change_2 + rental_car_2) * poisson.pmf(rental_car_2, POISSON_LAMBDA_RENTAL_CARS_2) * poisson.pmf(arrive_car_2, POISSON_LAMBDA_RETURN_CARS_2)

        return R_1 + R_2
        
    def __str__(self) -> str:
        return f"s_{self.state_1}_{self.state_2}: V={self.V}, R={self.R}, PI={self.PI}"

    def get_max_PI(self):
        max_value = max(self.PI.values())
        return [action for action, value in self.PI.items() if value == max_value]



class StateGrid:

    def __init__(self, size: int):
        self.size = size + 1
        self.states = {}
        self._create_grid()
    
    def _create_grid(self):
        for i in range(self.size):
            for j in range(self.size):
                self.states[f"s_{i}_{j}"] = State(f"s_{i}_{j}", i, j)
    
    def get_V(self):
        V = pd.DataFrame(index=range(self.size), columns=range(self.size))
        for i in range(self.size):
            for j in range(self.size):
                V.loc[i, j] = self.states[f"s_{i}_{j}"].V
        # order index descending
        # V = V.sort_index(axis=0, ascending=False)
        return V

    def get_R(self):
        R = pd.DataFrame(index=range(self.size), columns=range(self.size))
        for i in range(self.size):
            for j in range(self.size):
                R.loc[i, j] = self.states[f"s_{i}_{j}"].R
        return R
    
    def update_V_single_state(self, state: State):
        """+5 is location 1 recieves 5 cars, -5 is location 1 send 5 cars"""
        cars_location_1, cars_location_2 = state.state_1, state.state_2
        new_V = 0.0
        for action in state.PI.keys():
            action_value = int(action)
            if action_value > 0:
                if (cars_location_2 >= action_value) and (cars_location_1 + action_value <= STATE_SIZE):
                    next_state = self.states[f"s_{cars_location_1 + action_value}_{cars_location_2 - action_value}"]
                    new_V += state.PI[action] * ((next_state.R - ACTION_COST * action_value) + GAMMA * next_state.V)
                else:
                    cars_to_transfer = min(cars_location_2, STATE_SIZE - cars_location_1)
                    next_state = self.states[f"s_{cars_location_1 + cars_to_transfer}_{cars_location_2 - cars_to_transfer}"]
                    new_V += state.PI[action] * ((next_state.R - ACTION_COST * cars_to_transfer) + GAMMA * next_state.V)
            elif action_value < 0:
                if (cars_location_1 >= -action_value) and (cars_location_2 - action_value <= STATE_SIZE):
                    next_state = self.states[f"s_{cars_location_1 + action_value}_{cars_location_2 - action_value}"]
                    new_V += state.PI[action] * ((next_state.R - ACTION_COST * -action_value) + GAMMA * next_state.V)
                else:
                    cars_to_transfer = min(cars_location_1, STATE_SIZE - cars_location_2)
                    next_state = self.states[f"s_{cars_location_1 - cars_to_transfer}_{cars_location_2 + cars_to_transfer}"]
                    new_V += state.PI[action] * ((next_state.R - ACTION_COST * cars_to_transfer) + GAMMA * next_state.V)
            elif action_value == 0:
                next_state = self.states[f"s_{cars_location_1}_{cars_location_2}"]
                new_V += state.PI[action] * ((next_state.R) + GAMMA * next_state.V)
        state.V = new_V

    def update_V_all_states(self):
        for state in self.states.values():
            self.update_V_single_state(state)

    def policy_evaluation(self):
        while True:
            DELTA: float = 0.0
            for state in self.states.values():
                current_V = state.V
                self.update_V_single_state(state)
                DELTA = max(DELTA, abs(state.V - current_V))
            if DELTA < THETA:
                break

    def _argmax(self,policy_dict: dict):
        max_value = max(policy_dict.values())  # Get the maximum value
        counter = 0
        for _, value in policy_dict.items():
            if value == max_value:
                counter += 1
        for action, value in policy_dict.items():
            if value == max_value:
                policy_dict[action] = 1.0 / counter
            else:
                policy_dict[action] = 0.0
        return policy_dict

    def update_PI_single_state(self, state: State):
        """+5 is location 1 recieves 5 cars, -5 is location 1 send 5 cars"""
        actions = state.PI.keys()
        for action in actions:
            action_value = int(action)
            if action_value > 0:
                if (state.state_2 >= action_value) and (state.state_1 + action_value <= STATE_SIZE):
                    next_state = self.states[f"s_{state.state_1 + action_value}_{state.state_2 - action_value}"]
                    state.PI[action] = ((next_state.R - ACTION_COST * action_value) + GAMMA * next_state.V)
                else:
                    cars_to_transfer = min(state.state_2, STATE_SIZE - state.state_1)
                    next_state = self.states[f"s_{state.state_1 + cars_to_transfer}_{state.state_2 - cars_to_transfer}"]
                    state.PI[action] = ((next_state.R - ACTION_COST * cars_to_transfer) + GAMMA * next_state.V)
            elif action_value < 0:
                if (state.state_1 >= -action_value) and (state.state_2 - action_value <= STATE_SIZE):
                    next_state = self.states[f"s_{state.state_1 + action_value}_{state.state_2 - action_value}"]
                    state.PI[action] = ((next_state.R - ACTION_COST * -action_value) + GAMMA * next_state.V)
                else:
                    cars_to_transfer = min(state.state_1, STATE_SIZE - state.state_2)
                    next_state = self.states[f"s_{state.state_1 - cars_to_transfer}_{state.state_2 + cars_to_transfer}"]
                    state.PI[action] = ((next_state.R - ACTION_COST * cars_to_transfer) + GAMMA * next_state.V)
            elif action_value == 0:
                next_state = self.states[f"s_{state.state_1}_{state.state_2}"]
                state.PI[action] = ((next_state.R) + GAMMA * next_state.V)
        state.PI = self._argmax(state.PI)
    
    def policy_improvement(self):
        policy_stable = True
        for state in self.states.values():
            old_action = state.PI.copy()
            self.update_PI_single_state(state)
            if old_action != state.PI:
                policy_stable = False
        return policy_stable
    
    def get_PI(self):
        pi = pd.DataFrame(index=range(self.size), columns=range(self.size), dtype=object)
        for i in range(self.size):
            for j in range(self.size):
                pi.at[i, j] = self.states[f"s_{i}_{j}"].get_max_PI()
        return pi

def check_reward_function(state_i: int, state_j: int):
    rentals_1 = poisson.rvs(POISSON_LAMBDA_RENTAL_CARS_1, size=10000)
    returns_1 = poisson.rvs(POISSON_LAMBDA_RETURN_CARS_1, size=10000)
    rentals_2 = poisson.rvs(POISSON_LAMBDA_RENTAL_CARS_2, size=10000)
    returns_2 = poisson.rvs(POISSON_LAMBDA_RETURN_CARS_2, size=10000)

    rewards = pd.DataFrame({'rentals_1': rentals_1, 'returns_1': returns_1, 'rentals_2': rentals_2, 'returns_2': returns_2})
    rewards['net_change_1'] = rewards['returns_1'] - rewards['rentals_1'] + state_i
    rewards['net_change_2'] = rewards['returns_2'] - rewards['rentals_2'] + state_j
    
    rewards['reward_1'] = rewards.apply(lambda x: EARNING_PER_CAR * x['rentals_1'] if x['net_change_1'] >= 0 else EARNING_PER_CAR * (x['net_change_1'] + x['rentals_1']), axis=1)
    rewards['reward_2'] = rewards.apply(lambda x: EARNING_PER_CAR * x['rentals_2'] if x['net_change_2'] >= 0 else EARNING_PER_CAR * (x['net_change_2'] + x['rentals_2']), axis=1)
    rewards['reward'] = rewards['reward_1'] + rewards['reward_2']
    return rewards    
if __name__ == "__main__":
    state_grid = StateGrid(size=STATE_SIZE)
    print("Initial V:")
    print(state_grid.get_V())
    print("Initial R:")
    print(state_grid.get_R())
    policy_stable = False
    epoch = 0
    while not policy_stable:
        state_grid.policy_evaluation()
        policy_stable = state_grid.policy_improvement()
        epoch += 1
        print(f"Epoch: {epoch}")
    print(f"Policy stable: {policy_stable}")
    print("Final Policy:")
    print(state_grid.get_PI())
    print("Final V:")
    print(state_grid.get_V())
