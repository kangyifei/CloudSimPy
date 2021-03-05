from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv

class rllib_gym(ExternalMultiAgentEnv):
    def run(self):


        pass

    def start_episode(self, episode_id=None, training_enabled=True):
        return super().start_episode(episode_id, training_enabled)

    def get_action(self, episode_id, observation_dict):
        return super().get_action(episode_id, observation_dict)

    def log_action(self, episode_id, observation_dict, action_dict):
        return super().log_action(episode_id, observation_dict, action_dict)

    def log_returns(self, episode_id, reward_dict, info_dict=None, multiagent_done_dict=None):
        return super().log_returns(episode_id, reward_dict, info_dict, multiagent_done_dict)

    def end_episode(self, episode_id, observation_dict):
        return super().end_episode(episode_id, observation_dict)