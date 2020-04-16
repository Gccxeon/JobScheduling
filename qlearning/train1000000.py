import sys
sys.path.append('..')
from env_test import env
from env_test import policies as baseline_policy
from q_learning import trainer as QTrainer
from q_learning import QLearner


q_learner = QLearner(env, 0.05, 0.99)
q_trainer = QTrainer(q_learner, 50)

q_trainer.train(episodes=1000000, report_interval=100) 