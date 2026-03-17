import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase
from .fheragentwrapper import FHERagentWrapper

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING
# import mss

class FHERagentClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        env_path = self._get_env_path_from_task(self.name, config)
        
        return lambda: FHERagentWrapper(
                env_path=env_path,
                scenario=self.name.lower(),
                num_envs=num_envs,
                continuous_actions=continuous_actions,
                seed=seed,
                device=device,
                max_steps=config.get("max_steps", 1000),
                width=config.get("width", 600),
                height=config.get("height", 600),
                time_scale=config.get("time_scale", 10.0),
                render=config.get("render", True),
                manual_metrics_size=config.get("manual_metrics_size", 5)
            )
    
    def _get_env_path_from_task(self, task_name: str, config: Dict) -> str:
        """根据任务名确定环境路径"""
        # 方式1：从配置文件读取
        if "env_path" in config:
            return config["env_path"]
        
        # 方式2：根据任务名映射路径
        path_mapping = {
            "MAPF": r"E:\Python_code\UnityProject\ml-agents-release_20_docs\Project\MAPF_EASY_MAZE_FOG\UnityEnvironment",
            "EASY_MAZE_FOG": r"E:\Python_code\UnityProject\ml-agents-release_20_docs\Project\EASY_MAZE_FOG_V2\UnityEnvironment",
            "EASY_MAZE_STATIC": r"E:\Python_code\UnityProject\ml-agents-release_20_docs\Project\MAPF_EASY_MAZE\UnityEnvironment",
            # 添加更多任务到路径的映射
        }
        
        return path_mapping.get(task_name, config.get("default_env_path", ""))
    
    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("max_steps", 1000)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        n_agents = env.n_agents
        return {"agents": [f"agent_{i}" for i in range(n_agents)]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.full_observation_spec_unbatched

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    @staticmethod
    def env_name() -> str:
        return "fheragent"


class FHERTask(Task):
    """Enum for Unity ML-Agents tasks."""
    
    EASY_MAZE_STATIC = None
    EASY_MAZE_FOG = None
    EASY_MAZE_FIRE = None
    EASY_MAZE_STATIC16 = None
    
    MIDDLE_MAZE_STATIC = None
    MIDDLE_MAZE_STATIC16 = None
    MIDDLE_MAZE_FIRE = None
    MIDDLE_MAZE_FOG = None

    MIDDLE_SCHOOL_STATIC = None
    MIDDLE_SCHOOL_FIRE = None
    MIDDLE_SCHOOL_FOG = None
    
    MIDDLE_OFFICE_STATIC = None
    MIDDLE_OFFICE_FIRE = None
    MIDDLE_OFFICE_FOG = None

    HARD_MAZE_STATIC = None
    HARD_MAZE_FIRE = None
    HARD_MAZE_FOG = None

    @staticmethod
    def associated_class():
        return FHERagentClass