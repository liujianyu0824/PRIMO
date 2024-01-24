# Progressive Induction for Multi-hop Open Rule Generation

## Code Structure
The source code is organized as follows:

```data```: contains multi-hops open rule dataset.

```checkpoint```: contains the model weight files.

```data_process```: contains the code for preprocessing dataset.

```trl```: contains the utility functions.

```gen_openrule.py```: contains the main executable code for genarating open rule. 

```inference_ppo_gpt.py```: contains the code for testing the performance of gpt after ppo. 

```inference_reward.py```: contains the code for testing the performance of reward model.

```iTrainingLogger.py```: contains the code for loading dataset.

```ppo_openrule.py```: contains the code for reinforcement learning of G_net and E_net.

```train_reward_model.py```: contains the code for training reward model.

## Commands

### requirements
```
pip install -r requirements.txt
```

#### Training RM
```
python train_reward_model.py
```

#### Training gpt with ppo
```
python ppo_openrule.py
```

#### Generating multi-hops open rule
```
python gen_openrule.py
```
