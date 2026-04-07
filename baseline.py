import os
import httpx
from openai import OpenAI

BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")


def run_baseline_agent():
    http = httpx.Client(base_url=BASE_URL, timeout=30)
    ai = OpenAI()

    tasks_resp = http.get("/tasks")
    tasks = tasks_resp.json()["tasks"]

    scores = {}
    for task in tasks:
        task_id = task["task_id"]
        reset_resp = http.post("/reset", json={"task_id": task_id})
        obs_text = reset_resp.json().get("text", "")

        actions = []
        for _ in range(task["max_steps"]):
            completion = ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an agent completing a task. Reply with only the action text, nothing else."},
                    {"role": "user", "content": f"Task observation: {obs_text}\nWhat is your action?"},
                ],
                max_tokens=50,
            )
            action_text = completion.choices[0].message.content.strip()
            actions.append(action_text)

            step_resp = http.post("/step", json={"action": {"text": action_text}})
            step_data = step_resp.json()
            obs_text = step_data.get("text", "")
            if step_data.get("done"):
                break

        grader_resp = http.post("/grader", json={"task_id": task_id, "actions": actions})
        scores[task_id] = grader_resp.json()["grader_score"]

    http.close()
    return scores


if __name__ == "__main__":
    import json
    print(json.dumps(run_baseline_agent()))
