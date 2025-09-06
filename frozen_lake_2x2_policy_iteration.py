THETA = 0.001
GAMMA = 0.1

states = ['s_0', 's_1', 's_2']

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

def update_V_s_0():
    new_V = PI_s_0['up'] * (R['s_0'] + GAMMA * V['s_0']) + \
            PI_s_0['right'] * (R['s_1'] + GAMMA * V['s_1']) + \
            PI_s_0['down'] * (R['s_2'] + GAMMA * V['s_2']) + \
            PI_s_0['left'] * (R['s_0'] + GAMMA * V['s_0'])
    return new_V

def update_V_s_1():
    new_V = PI_s_1['up'] * (R['s_1'] + GAMMA * V['s_1']) + \
            PI_s_1['right'] * (R['s_1'] + GAMMA * V['s_1']) + \
            PI_s_1['down'] * (1.0) + \
            PI_s_1['left'] * (R['s_0'] + GAMMA * V['s_0'])
    return new_V

def update_V_s_2():
    new_V = PI_s_2['up'] * (R['s_0'] + GAMMA * V['s_0']) + \
            PI_s_2['right'] * (1.0) + \
            PI_s_2['down'] * (R['s_2'] + GAMMA * V['s_2']) + \
            PI_s_2['left'] * (R['s_2'] + GAMMA * V['s_2'])
    return new_V

def argmax(policy_dict: dict):
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



if __name__ == "__main__":
    policy_stable = False
    while not policy_stable:
        ## Policy Evaluation
        while True:

            DELTA: float = 0.0

            # state 0
            current_V = V['s_0']
            V['s_0'] = update_V_s_0()
            DELTA = max(DELTA, abs(V['s_0'] - current_V))

            # state 1
            current_V = V['s_1']
            V['s_1'] = update_V_s_1()
            DELTA = max(DELTA, abs(V['s_1'] - current_V))

            # state 2
            current_V = V['s_2']
            V['s_2'] = update_V_s_2()
            DELTA = max(DELTA, abs(V['s_2'] - current_V))

            print(f"DELTA: {DELTA}")
            print(f"V: {V}")

            if DELTA < THETA:
                break

        ## Policy Improvement
        policy_stable = True
        for state in states:
            if state == 's_0':
                old_action = PI_s_0

                PI_s_0 = {
                    'up': R['s_0'] + GAMMA * V['s_0'],
                    'right': R['s_1'] + GAMMA * V['s_1'],
                    'down': R['s_2'] + GAMMA * V['s_2'],
                    'left': R['s_0'] + GAMMA * V['s_0'],
                }
                
                PI_s_0 = argmax(PI_s_0)
                print(f"PI_s_0: {PI_s_0}")

                if old_action != PI_s_0:
                    policy_stable = False
                
            elif state == 's_1':
                old_action = PI_s_1

                PI_s_1 = {
                    'up': R['s_1'] + GAMMA * V['s_1'],
                    'right': R['s_1'] + GAMMA * V['s_1'],
                    'down': 1.0,
                    'left': R['s_0'] + GAMMA * V['s_0'],
                }
                
                PI_s_1 = argmax(PI_s_1)
                print(f"PI_s_1: {PI_s_1}")

                if old_action != PI_s_1:
                    policy_stable = False
                
            elif state == 's_2':
                old_action = PI_s_2

                PI_s_2 = {
                    'up': R['s_0'] + GAMMA * V['s_0'],
                    'right': 1.0,
                    'down': R['s_2'] + GAMMA * V['s_2'],
                    'left': R['s_2'] + GAMMA * V['s_2'],
                }
                

                PI_s_2 = argmax(PI_s_2)
                print(f"PI_s_2: {PI_s_2}")

                if old_action != PI_s_2:
                    policy_stable = False

    if policy_stable:
        print("Policy stable")
        print(f"PI_s_0: {PI_s_0}")
        print(f"PI_s_1: {PI_s_1}")
        print(f"PI_s_2: {PI_s_2}")
        print(f"V: {V}")

    print(f"Policy stable: {policy_stable}")
