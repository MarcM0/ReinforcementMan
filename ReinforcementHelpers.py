import gym
from Constants import *
from stable_baselines3 import A2C
import os
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

def reinforcementLoad():
    #create environment
    env = gym.make(environmentName)
    env.reset()

    #algo
    model = A2C.load(modelCheckpoint, env=env)

    #TODO
    episodes = 5

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            print(rewards)
            
def reinforcementTrain():
    #create folders
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    #create environment
    env = gym.make(environmentName)
    # Separate evaluation env
    eval_env = gym.make(environmentName)
    eval_env.reset()
    env.reset()

    #algo
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

    #callbacks
    eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir+"/"+neatHyperparams["modelName"]+"/best_model",
                                log_path=logdir, eval_freq=neatHyperparams["evalFreq"])
    checkpoint_callback = CheckpointCallback(save_freq=neatHyperparams["checkpointFreq"], save_path=models_dir+"/"+neatHyperparams["modelName"])
    callback = CallbackList([checkpoint_callback, eval_callback])


    #training loop
    model.learn(total_timesteps=neatHyperparams["totalSteps"], reset_num_timesteps=False, tb_log_name=neatHyperparams["modelName"],callback=callback)

