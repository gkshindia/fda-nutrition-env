from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from env.models import FDAAction, FDAObservation, FDAState


class FDAEnv(EnvClient[FDAAction, FDAObservation, FDAState]):
    def _step_payload(self, action: FDAAction):
        return action.model_dump()

    def _parse_result(self, payload):
        obs_data = payload.get("observation", payload) if payload else {}
        reward = payload.get("reward")
        done = payload.get("done", False)
        obs = FDAObservation(**obs_data) if obs_data else FDAObservation()
        return StepResult(observation=obs, reward=reward if reward is not None else obs.reward, done=done or obs.done)

    def _parse_state(self, payload):
        return FDAState(**payload) if payload else FDAState()
