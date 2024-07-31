# 07/29/2024
## 12:45 PM

Did a fresh-install of python 3.9.5 due to some issues with 3.9.11
Started training with layers size of 2048, 2048, 1024.
Avaraging 6,803 sps with 3 epochs.

## 12:56 PM

AND IT BROKE AGAIN

```
LEARNING LOOP ENCOUNTERED AN ERROR

Traceback (most recent call last):
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\rlgym_ppo\learner.py", line 225, in learn
    self._learn()
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\rlgym_ppo\learner.py", line 257, in _learn
    experience, collected_metrics, steps_collected, collection_time = self.agent.collect_timesteps(
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\rlgym_ppo\batched_agents\batched_agent_manager.py", line 110, in collect_timesteps
    self._send_actions()
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\utils\_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\rlgym_ppo\batched_agents\batched_agent_manager.py", line 203, in _send_actions
    actions, log_probs = self.policy.get_action(inference_batch)
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\rlgym_ppo\ppo\discrete_policy.py", line 59, in get_action
    action = torch.multinomial(probs, 1, True)
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
```

Ok so, apparently this is caused by the reward returning nan.
Should be fixed now.

## 01:04 PM

```
C:\Users\Jueferu\Desktop\Fresh\rewards\BeginnerReward.py:33: RuntimeWarning: invalid value encountered in divide
  player_velocity = player_velocity / np.linalg.norm(player_velocity)
```
hm, im sure this won't cause problems later
:clueless:

## 01:39 PM

Ok. Stopped training at 22k timesteps (22m checkpoints)

Time to test it in rlbot

## 01:57 PM

It's not working lmao
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x107 and 169x2048)
```

what i assume is wrong here is my config
HOWEVER

OBS_SIZE(169) and POLICY_LAYER_SIZES(2048, 2048, 1024) have been set correctly
```
Received request for env shapes, returning:
- Observations shape: (169,)       
- Number of actions: 90.0
- Action space type: 0.0 (Discrete)
```

## 2:19 PM

Turns out that the default bot.py forces the observation to be 1v1 only
So I just had to remove the code for that

Also learned that the normal threshhold for when the bot should start chasing the ball is at 1b steps
We currently sit at 3m, so we gotta let it cook now

## 2:24 PM

1. More processes != more ram usage. Can support up to 32 processes now, giving us about 10k avarage sps.
2. THE BUG IS BACK

```
LEARNING LOOP ENCOUNTERED AN ERROR

Traceback (most recent call last):
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\serialization.py", line 652, in save
    _save(obj, opened_zipfile, pickle_module, pickle_protocol, _disable_byteorder_record)
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\serialization.py", line 886, in _save
    zip_file.write_record(name, storage, num_bytes)
RuntimeError: [enforce fail at inline_container.cc:783] . PytorchStreamWriter failed writing file data/7: file write failed

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\rlgym_ppo\learner.py", line 225, in learn
    self._learn()
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\rlgym_ppo\learner.py", line 325, in _learn
    self.save(self.agent.cumulative_timesteps)
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\rlgym_ppo\learner.py", line 417, in save
    self.ppo_learner.save_to(folder_path)
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\rlgym_ppo\ppo\ppo_learner.py", line 246, in save_to
    torch.save(
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\serialization.py", line 653, in save
    return
  File "C:\Users\Jueferu\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\serialization.py", line 499, in __exit__
    self.file_like.write_end_of_file()
RuntimeError: [enforce fail at inline_container.cc:603] . unexpected pos 20329856 vs 20329748
```

turns out this bug is caused by lack of drive space
which also means my bot is BIG AS FUCKK
while 10GB is very low normally, i would expect it to at least hold a little bit

time to move to linux training

# 07/30/2024
## 7:16 AM

Could not move to linux due to issues with python3.9 not installing github packages
Also restarted the bot

I left it training during the night, we currently sit at 300M timesteps.
Time to see what progress it has made in RLBot.

It's just spinning around dude

Trying to fix the lookup act didn't work
I just have to assume the bot hasn't trained enough

## 1:06 PM
Restared the bot again lmaoo :crying:
The reward that i was training him on for 200M steps was wrong

Anyways, I fixed it.

## 4:49 PM
Did a major overhaul of the bot and opened this github page (hello reader)

Guess what? Of course you knew: BOT RESTART

## 5:17 PM
Finished setting up the github
Time to train

Reward function returning nan?
OH HAHAHA
very funny
i logged it and it fixed itself
very funny python

## 6:09 PM
20m steps reached
time to test in rlbot

## 6:20 PM
rocket league keeps crashing for some reason, will try without bakesmod

IT WORKS
IT'S ALIVEE YESS
(it's still bad lmao)

time to train more

# 07/31/2024

So we have reached 25m steps AND I JUST REALISED I HAVE BEEN TRAINING THE BOT ON 1 TICK SKIP
:sob:

that's why i only reached 25m now
lmaoo

## 4:27 PM

We have reached 27m steps
I'll test in rlbot just to make sure

https://youtu.be/siOV2oJQGFI
