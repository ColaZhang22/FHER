import torch
import numpy as np
from typing import Optional, Dict, Any, List
from torchrl.envs import EnvBase
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedContinuousTensorSpec, BoundedTensorSpec, DiscreteTensorSpec
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import atexit 
from benchmarl.utils import DEVICE_TYPING


class FHERagentWrapper(EnvBase):
    """Unity ML-Agents environment wrapper for BenchMARL"""
    
    def __init__(
        self,
        env_path: str,
        scenario: str = "easymazefire",
        num_envs: int = 1,
        continuous_actions: bool = True,
        seed: Optional[int] = None,
        device: DEVICE_TYPING = "cpu",
        max_steps: int = 1024,
        worker_id: int = 0, 
        **kwargs,
    ):  
        print(f"kwargs: {kwargs}")
        self.env_path = env_path
        self.scenario = scenario
        self.num_envs = num_envs  # Unity环境通常是单环境
        self.continuous_actions = continuous_actions
        self.max_steps = max_steps
        self.worker_id = worker_id 
        self._step_count = 0
        self.manual_metrics_size = kwargs.get("manual_metrics_size", 4)
        # if True:
        #     # self.manual_metrics_size = 4
        #     self.manual_metrics_size = 5
        # Initialize Unity Environment
        self._init_unity_env(seed, **kwargs)
        super().__init__(device=device)
        # Set Observation Space and Action Space 
        self._make_specs()

    def _init_unity_env(self, seed: Optional[int], **kwargs):
        # Render Configuration
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(
            width=kwargs.get("width", 800),
            height=kwargs.get("height", 800),
            time_scale=kwargs.get("time_scale", 10.0)
        )
        # Create Environment
        max_retries = 20
        for i in range(max_retries):
            try:
                current_worker_id = self.worker_id + i 
                # print(f"🔄 Trying to connect to Unity with worker_id={current_worker_id}")
                self.env = UnityEnvironment(
                    file_name=self.env_path,
                    seed=seed or 1,
                    side_channels=[channel],
                    worker_id=current_worker_id,
                    no_graphics=not kwargs.get("render", True)
                )
                print(f"Connected to Unity with worker_id={current_worker_id}")
                break
            except Exception as e:
                current_worker_id += 1
                if i == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to connect to Unity after {max_retries} attempts. ")
                continue
                

        # Reset Environment and Get Specs
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        
        # Environment and Agent Info
        self.n_agents = len(decision_steps)
        self.obs_dim = self.spec.observation_specs[0].shape[0] - self.manual_metrics_size
        
        if self.spec.action_spec.is_continuous():
            self.action_dim = self.spec.action_spec.continuous_size 
        else:
            self.action_dim = len(self.spec.action_spec.discrete_branches)
            
        print(f"--------Unity Environment Info:--------")
        print(f"  Agents: {self.n_agents}")
        print(f"  Obs Dim: {self.obs_dim}")
        print(f"  Action Dim: {self.action_dim}")
        print(f"  Continuous Actions: {self.continuous_actions}")
        print(f"  Manual Metrics Size: {getattr(self, 'manual_metrics_size', 'N/A')}")
        print(f"---------------------------------------")

    def _make_specs(self):
         # 观察空间规格
       
        observation_spec = Composite({
            "agents": Composite({
                "observation": UnboundedContinuousTensorSpec(
                    shape=(self.n_agents, self.obs_dim),
                    dtype=torch.float32,
                )
            }, shape=(self.n_agents,) )
        })
        
        # 动作空间规格
        low = torch.full((self.n_agents, self.action_dim), -1.0, dtype=torch.float32)
        high = torch.full((self.n_agents, self.action_dim), 1.0, dtype=torch.float32)
        # print(f"low shape: {low.shape}, values: {low}")
        # print(f"high shape: {high.shape}, values: {high}")
        action_spec = Composite({
            "agents": Composite({
                "action": BoundedTensorSpec(
                    low=low,
                    high=high,
                    shape=(self.n_agents, self.action_dim),
                    dtype=torch.float32,
                )
            }, shape=(self.n_agents,) )
        })
        test_spec = action_spec["agents", "action"]
    
        reward_spec = Composite({
            "agents": Composite({
                "reward": UnboundedContinuousTensorSpec(
                    shape=(self.n_agents,1),
                    dtype=torch.float32,
                ),
            }, shape=(self.n_agents,))
        })

        
        # ✅ 完成状态规格 - 使用不同的结构  
        single_done_spec = Composite({
            "agents": Composite({
                "done_list": DiscreteTensorSpec(
                    n=2,
                    shape=(self.n_agents,1),
                    dtype=torch.bool,
                )
            }, shape=(self.n_agents,))
        })
        
        done_spec = Composite({
                "done": DiscreteTensorSpec(
                    n=2,
                    shape=(1,),
                    dtype=torch.bool,
                )
            }, shape=(1,))
     

        self.full_observation_spec = observation_spec
        self.full_action_spec = action_spec
        self.full_reward_spec = reward_spec
        self.full_done_spec = done_spec

    def _reset(self, tensordict: TensorDict = None, **kwargs) -> TensorDict:
        """重置环境"""
        self.env.reset()
        self._step_count = 0
        self.agents_finished = np.zeros(self.n_agents, dtype=bool)

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        # 获取观察
        obs = decision_steps.obs[0][:, :self.obs_dim] # shape: (n_agents, obs_dim)
        dones = torch.zeros((self.n_agents,1), dtype=torch.bool, device=self.device)
        global_done = torch.zeros(1, dtype=torch.bool, device=self.device)
        global_terminated = torch.zeros(1, dtype=torch.bool, device=self.device)
        global_truncated = torch.zeros(1, dtype=torch.bool, device=self.device)
        manual_metrics = torch.zeros((1,self.n_agents), dtype=torch.bool, device=self.device)

        
        # 转换为TensorDict
        return TensorDict({
            "agents": {
                "observation": torch.tensor(obs, dtype=torch.float32, device=self.device),
                "reward": torch.zeros((self.n_agents,1), dtype=torch.float32, device=self.device),
                "done_list": dones, 
                "done": dones     
            },
            "manual_metrics": manual_metrics, 
            "done": global_done,
            "terminated": global_terminated,
            "truncated": global_truncated
        }, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """执行一步动作"""
        actions = tensordict["agents"]["action"]
        
        # 转换动作格式
        actions_np = actions.cpu().numpy()
        
    
        # 创建Unity动作
        action_tuple = ActionTuple()
        if self.continuous_actions:
            action_tuple.add_continuous(actions_np)
        else:
            action_tuple.add_discrete(actions_np)
        
     
        # 执行动作
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()
        
        # 获取结果
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        
        unity_total_dim = self.obs_dim + self.manual_metrics_size
        full_obs = np.zeros((self.n_agents, unity_total_dim), dtype=np.float32)
        rewards = np.zeros(self.n_agents, dtype=np.float32)
    
        # Combine Decision Steps and Terminal Steps
        for agent_id in decision_steps.agent_id:
            if agent_id < self.n_agents:
                idx = decision_steps.agent_id_to_index[agent_id]
                full_obs[agent_id] = decision_steps.obs[0][idx]
                rewards[agent_id] = decision_steps.reward[idx]

        for agent_id in terminal_steps.agent_id:
            if agent_id < self.n_agents:
                idx = terminal_steps.agent_id_to_index[agent_id]
                full_obs[agent_id] = terminal_steps.obs[0][idx]
                rewards[agent_id] = terminal_steps.reward[idx]

        # Observation t+1
        next_obs = full_obs[:, :self.obs_dim] # shape: (n_agents, obs_dim)
        if True:
            metrics_raw =  full_obs[:, -self.manual_metrics_size:] 
            manual_metrics = np.mean(metrics_raw, axis=0) # Average over agents
            # print(f"Manual Metrics Averaged: {manual_metrics} \n")
        
        

        # 处理完成状态（根据你的逻辑：reward == 5 表示完成）
        is_success = (rewards >= 4.9)
        self.agents_finished = self.agents_finished | is_success
        
        all_success = self.agents_finished.all()
        self._step_count += 1
        is_timeout = self._step_count >= self.max_steps
     
        
        agents_done_np = self.agents_finished | is_timeout
        global_done_bool = all_success or is_timeout
        global_terminated_bool = global_done_bool
        global_truncated_bool = False 
  
        # print(f"Step {self._step_count}: Rewards: {rewards}, Is Success: {agents_done_np}, all success {global_done_bool} Timeout: {is_timeout} \n")
        
        # 转换为TensorDict
        return TensorDict({
            "agents": {
                "observation": torch.tensor(next_obs, dtype=torch.float32, device=self.device),
                "reward": torch.tensor(rewards, dtype=torch.float32, device=self.device).view(self.n_agents, 1),
                "done_list": torch.tensor(agents_done_np, dtype=torch.bool, device=self.device).view(self.n_agents, 1),
               },
            "manual_metrics": torch.tensor(manual_metrics, dtype=torch.float32, device=self.device).view(1, -1),
            "done": torch.tensor(global_done_bool, dtype=torch.bool, device=self.device),
            "terminated": torch.tensor(global_terminated_bool, dtype=torch.bool, device=self.device),
            "truncated": torch.tensor(global_truncated_bool, dtype=torch.bool, device=self.device),
        }, batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed: Optional[int]):
        """设置随机种子"""
        self.seed = seed
        # if seed is not None:
        #     # Unity环境需要重新创建才能设置种子
        #     self.env.close()
        #     self._init_unity_env(seed)

    def close(self, **kwargs): 
        """关闭环境 (支持 **kwargs 以兼容 TorchRL)"""
        if hasattr(self, 'env'):
            try:
             
                if hasattr(self.env, '_close'):
                    atexit.unregister(self.env._close)
                self.env.close()
                # print("Closing Unity Environment...成功关闭")
            except Exception:
                pass

    def render(self):
        import mss
        with mss.mss() as sct:
            # 获取主屏幕
            monitor = sct.monitors[1]
            # 截图
            img = np.array(sct.grab(monitor))
            # 转换格式: BGRA -> RGB
            img = img[:, :, :3][:, :, ::-1]
            
            # 可选：调整大小以减小视频体积
            # import cv2
            # img = cv2.resize(img, (640, 480))
            return img
        
    @property
    def agents(self) -> List[str]:
        """返回智能体名称列表"""
        return [f"agent_{i}" for i in range(self.n_agents)]

    def __del__(self):
        """析构函数"""
        self.close()