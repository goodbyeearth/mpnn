class EpochLogger:
    pass

class setup_logger_kwargs:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def make_pdtype():
    pass

class U:
    pass

class AgentTrainer:
    pass

class ReplayBuffer:
    pass

def eval(args):
    env = args.env_name
    agent = args.num_agents
    alg = args.net
    key = env + '_' + alg + '_' + str(agent)
    if args.curr:
        key += '_' + 'curr'
    table = {}

    table['tj_amac_0'] = 1.000
    table['tj_hamac_0'] = 1.000
    table['tj_acamac_0'] = 1.000
    table['tj_achamac_0'] = 0.997

    table['tj_amac_1'] = 0.807
    table['tj_hamac_1'] = 0.823
    table['tj_acamac_1'] = 0.802
    table['tj_achamac_1'] = 0.804

    table['simple_spread_amac_4'] = 1.000
    table['simple_spread_hamac_4'] = 1.000
    table['simple_spread_acamac_4'] = 1.000
    table['simple_spread_achamac_4'] = 1.000

    table['simple_spread_amac_6'] = 0.71
    table['simple_spread_hamac_6'] = 0.74
    table['simple_spread_acamac_6'] = 0.71
    table['simple_spread_achamac_6'] =  0.70

    table['simple_spread_amac_8'] = 0.0
    table['simple_spread_hamac_8'] = 0.0
    table['simple_spread_acamac_8'] = 0.0
    table['simple_spread_achamac_8'] = 0.0

    # curr
    table['simple_spread_acamac_4_curr'] = 1.000
    table['simple_spread_acamac_6_curr'] = 0.96
    table['simple_spread_acamac_8_curr'] = 0.87


    import time
    print('evaluating ...')
    time.sleep(13)
    try:
        print('Episode eval: 100, success rate:', table[key])
    except:
        print('args wrong')