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
STATE_SIZE: int = 5
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
        self.state_up = None
        self.state_down = None
        self.state_left = None
        self.state_right = None
        self.V = 0.0
        self.PI = self._init_PI()
        self.R = self._calculate_R()
    
    def _init_PI(self):
        PI = {}
        for action in ACTIONS:
            PI[action] = 1.0 / len(ACTIONS)
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


class StateGrid:

    def __init__(self, size: int):
        self.size = size + 1
        self.states = {}
        self._create_grid()
    
    def _create_grid(self):
        for i in range(self.size):
            for j in range(self.size):
                self.states[f"s_{i}_{j}"] = State(f"s_{i}_{j}", i, j)
        self._create_connections()
    
    def _create_connections(self):
        for i in range(self.size):
            for j in range(self.size):
                if i > 0:
                    self.states[f"s_{i}_{j}"].state_up = self.states[f"s_{i-1}_{j}"]
                if i < self.size - 1:
                    self.states[f"s_{i}_{j}"].state_down = self.states[f"s_{i+1}_{j}"]
                if j > 0:
                    self.states[f"s_{i}_{j}"].state_left = self.states[f"s_{i}_{j-1}"]
                if j < self.size - 1:
                    self.states[f"s_{i}_{j}"].state_right = self.states[f"s_{i}_{j+1}"]
    
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
        for action in ACTIONS:
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
    # print(state_grid.states['s_0_0'])
    print("Initial V:")
    print(state_grid.get_V())
    print("Initial R:")
    print(state_grid.get_R())
    # print('update V from s_0_0')
    # state_grid.update_V_single_state(state_grid.states['s_0_0'])
    # print(state_grid.states['s_0_0'])
    print('update V from all states')
    state_grid.update_V_all_states()
    print(state_grid.get_V())
    # rewards_0_0 = check_reward_function(0, 0)
    # rewards_0_20 = check_reward_function(0, 20)
    # rewards_20_20 = check_reward_function(20, 20)
    # print(rewards_0_0['reward'].mean())
    # print(rewards_0_20['reward'].mean())
    # print(rewards_20_20['reward'].mean())