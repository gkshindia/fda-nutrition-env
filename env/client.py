from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from env.models import FDAAction, FDAObservation, FDAState


class FDAEnv(EnvClient[FDAAction, FDAObservation, FDAState]):
    def _step_payload(self, action: FDAAction):
        return action.model_dump()

    def _parse_result(self, payload):
        obs = FDAObservation(**payload) if payload else FDAObservation()
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def _parse_state(self, payload):
        return FDAState(**payload) if payload else FDAState()
