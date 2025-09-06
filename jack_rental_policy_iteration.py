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
        rental_cars_1 = range(0, STATE_SIZE + 1)
        arrive_cars_1 = range(0, STATE_SIZE + 1)
        rental_cars_2 = range(0, STATE_SIZE + 1)
        arrive_cars_2 = range(0, STATE_SIZE + 1)
        for rental_car_1 in rental_cars_1:
            for arrive_car_1 in arrive_cars_1:
                state_1 = self.state_1 - rental_car_1 + arrive_car_1
                if state_1 >= 0:
                    R_1 += EARNING_PER_CAR * rental_car_1 * poisson.pmf(rental_car_1, POISSON_LAMBDA_RENTAL_CARS_1)
                elif state_1 < 0:
                    R_1 += EARNING_PER_CAR * self.state_1 * poisson.pmf(rental_car_1, POISSON_LAMBDA_RENTAL_CARS_1)

        for rental_car_2 in rental_cars_2:
            for arrive_car_2 in arrive_cars_2:
                state_2 = self.state_2 - rental_car_2 + arrive_car_2
                if state_2 >= 0:
                    R_2 += EARNING_PER_CAR * rental_car_2 * poisson.pmf(rental_car_2, POISSON_LAMBDA_RENTAL_CARS_2)
                elif state_2 < 0:
                    R_2 += EARNING_PER_CAR * self.state_2 * poisson.pmf(rental_car_2, POISSON_LAMBDA_RENTAL_CARS_2)

        return R_1 + R_2

    # def get_state_according_to_action(self, action: str):
    #     action = int(action)
    #     state_i = self.index_i + action
    #     state_j = self.index_j
    #     if state_i < 0 or state_i > self.size or state_j < 0 or state_j > self.size:
    #         return None
    #     return self.states[f"s_{state_i}_{state_j}"]





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
    

    
if __name__ == "__main__":
    state_grid = StateGrid(size=STATE_SIZE)
    print(state_grid.get_V())
    print(state_grid.get_R())