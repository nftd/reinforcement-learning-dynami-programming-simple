THETA = 0.001
GAMMA = 0.5

states = ['s_0', 's_1', 's_2', 's_3']

V = {
    's_0': 0.0,
    's_1': 0.0,
    's_2': 0.0
}

PI_s_0 = {
    'up': 0.25,
    'down': 0.25,
    'left': 0.25,
    'right': 0.25
}

PI_s_1 = {
    'up': 0.25,
    'down': 0.25,
    'left': 0.25,
    'right': 0.25
}

PI_s_2 = {
    'up': 0.25,
    'down': 0.25,
    'left': 0.25,
    'right': 0.25
}

R = {
    's_0': 0.0,
    's_1': 0.0,
    's_2': 0.0,
    's_3': 1.0,
}



if __name__ == "__main__":
    ## Start
    DELTA: float = 0.0

    # state 0
    current_V = V['s_0']
    new_V = PI_s_0['up'] * (R['s_0'] + GAMMA * V['s_0']) + \
            PI_s_0['right'] * (R['s_1'] + GAMMA * V['s_1']) + \
            PI_s_0['down'] * (R['s_2'] + GAMMA * V['s_2']) + \
            PI_s_0['left'] * (R['s_0'] + GAMMA * V['s_0'])
    DELTA = max(DELTA, abs(new_V - current_V))
