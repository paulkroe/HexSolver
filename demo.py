import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Agent.agent as Agent
import Game.Hex as Hex
import Agent.Networks.CNN as CNN
from Training.SelfPlayEnvironment import SelfPlayEnvironment, EvaluationEnvironment, train_agents
import Training.wandb_logger as wandb_logger

model1 = CNN.AlphaZeroNetwork(Hex.SIZE, Hex.SIZE)
model2 = CNN.AlphaZeroNetwork(Hex.SIZE, Hex.SIZE)
model3 = CNN.AlphaZeroNetwork(Hex.SIZE, Hex.SIZE)
model4 = CNN.AlphaZeroNetwork(Hex.SIZE, Hex.SIZE)
model5 = CNN.AlphaZeroNetwork(Hex.SIZE, Hex.SIZE)
    
agent1 = Agent.Agent(model1, "agent1")
agent2 = Agent.Agent(model2, "agent2")
agent3 = Agent.Agent(model3, "agent3")
agent4 = Agent.Agent(model4, "agent4")
agent5 = Agent.Agent(model5, "agent5")

agent1.logger = wandb_logger.WandBLogger(model=agent1.model, enabled=True)


agent_pool = [agent1, agent2, agent3, agent4, agent5]
train_agents(agent_pool=agent_pool, num_games=500)
EvaluationEnvironment(agent_pool)()