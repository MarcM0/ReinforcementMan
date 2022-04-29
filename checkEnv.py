from stable_baselines3.common.env_checker import check_env
from Main import pacEnv
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#set checkenv mode to true in constants
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
env = pacEnv()
env.run()
# It will check your custom environment and output additional warnings if needed
check_env(env)

episodes = 50

for episode in range(episodes):
    print("episode1-",episode)
    done = False
    obs = env.reset()
    print("episode2-",episode)
    while not done:#not done:
        random_action = env.action_space.sample()
        #print("action",random_action)
        obs, reward, done, info = env.step(random_action)
        #print('reward',reward)
    print("episode3-",episode)