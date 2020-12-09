To start the training loop, run:
    cd floyd_net
    python dqn.py
    
Code directory:
    common: backend API functions to translate graphs between formats
    pytorch_structure2vec: pytorch implementation of S2V
    floyd_net: 
        dqn.py: code for main training loop (i.e. get actions, update states, calculate loss, etc.)
        q_net.py: definition of DQN and full network pipeline
        message.py: random graph generation and efficiency calculation
        