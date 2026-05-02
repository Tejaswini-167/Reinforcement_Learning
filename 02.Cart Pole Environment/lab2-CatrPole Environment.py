import gymnasium as gym
import time
from PIL import Image
import os

env=gym.make("CartPole-v1",render_mode="rgb_array")

env.reset()
frame = env.render()

print(env.observation_space)
print(env.action_space)

n_epsds = 20
screenshot_saved = False
for n in range(n_epsds):    
    env.reset()
    steps = 10
    ret = 0
    for i in range(steps):
        a = env.action_space.sample()
        (ns, reward, done, info, prob) = env.step(a)
        ret = ret + reward
        frame = env.render()
        
        # Save first frame as screenshot
        if not screenshot_saved and frame is not None:
            os.makedirs("../Screenshots", exist_ok=True)
            img = Image.fromarray(frame)
            img.save("../Screenshots/lab2-output.png")
            screenshot_saved = True
        
        if done:
            break
    if n % 5 == 0:
        print("Return of this episode {0} is {1}".format(n, ret))
        

