import os

from stable_baselines3 import PPO


from connectx.utils import get_agent, TqdmCallback, get_agentV2, get_agentV3, save_model_data


class BaseModel:
    def __init__(self, env, model_name, model_params, policy="MlpPolicy", new_model=True, print_model=True):
        self.model_name = model_name
        self.model_params = model_params
        self.base_policy = policy

        self.log_dir = "logs/"
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_dir = f"saved_models/{model_name}"
        os.makedirs(self.log_dir, exist_ok=True)

        self.learner = PPO(self.base_policy, env, verbose=0, tensorboard_log=self.log_dir, **self.model_params)

        if print_model:
            print(self.learner.policy)
        if new_model is True:
            save_model_data(model_name, model_params, self.learner)

    def load_model_version(self, env, version):
        self.learner = PPO(self.base_policy, env, verbose=0, tensorboard_log=self.log_dir, **self.model_params)
        self.learner = self.learner.load(f"{self.model_dir}/{version}", env=env)

    def learn(self, timesteps):
        self.learner.learn(
            total_timesteps=timesteps,
            tb_log_name=self.model_name,
            callback=TqdmCallback(timesteps),
            reset_num_timesteps=False,
        )

    def save(self, version):
        self.learner.save(f"{self.model_dir}/{version}")

    def get_agent(self):
        return get_agent(self.learner)


class BaseModelV2(BaseModel):
    def get_agent(self):
        return get_agentV2(self.learner)

    def get_agentV3(self):
        return get_agentV3(self.learner)


class BaseModelV3(BaseModel):
    def get_agent(self):
        return get_agentV3(self.learner)
