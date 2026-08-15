"""Runnable demo of the sandbox: `uv run python -m src.llm.tools.meta.sandbox`.

Shows the moves that matter — token-free polling with `sleep()`, concurrent
fan-out with `parallel(...)`, and an `import` attempt being rejected —
against mock tools, with no LLM involved.
"""

import asyncio
import random
import time

from .interpreter import Interpreter

JOBS: dict[str, dict] = {}


async def add_task(url: str) -> str:
    job_id = f"job_{random.randint(1000, 9999)}"
    JOBS[job_id] = {"status": "running", "started": time.time(), "url": url}
    return job_id


async def get_job_status(job_id: str) -> str:
    job = JOBS.get(job_id)
    if not job:
        return "not_found"
    # pretend jobs finish after 0.5s
    if time.time() - job["started"] > 0.5:
        job["status"] = "success"
    return job["status"]


async def send_notification_to_admin(admin_id: int) -> dict:
    await asyncio.sleep(0.3)  # simulate network latency
    return {"admin_id": admin_id, "delivered": True}


TOOLS = {
    "add_task": add_task,
    "get_job_status": get_job_status,
    "send_notification_to_admin": send_notification_to_admin,
}

LLM_CODE = """
try:
    job_id = add_task("https://www.example.com")
    functions_responses = None
    if job_id:
        while get_job_status(job_id) == "running":
            sleep(0.2)
        job_status = get_job_status(job_id)
        print(f"job {job_id} finished: {job_status}")
        if job_status == "success":
            functions_responses = parallel(
                send_notification_to_admin(1),
                send_notification_to_admin(2),
                send_notification_to_admin(3),
                send_notification_to_admin(4),
            )
    return functions_responses
except Exception as e:
    return str(e)
"""

BLOCKED_CODE = """
import os
os.system("rm -rf /")
"""


async def main():
    interp = Interpreter(TOOLS)

    t0 = time.time()
    out = await interp.run(LLM_CODE)
    elapsed = time.time() - t0
    print("== main scenario ==")
    print("result:", out["result"])
    print("logs:", out["logs"])
    print("error:", out["error"])
    print(f"elapsed: {elapsed:.2f}s " f"(4 notifications x 0.3s ran in parallel, not 1.2s serial)")

    print("\n== blocked scenario ==")
    out2 = await interp.run(BLOCKED_CODE)
    print("error:", out2["error"])


if __name__ == "__main__":
    asyncio.run(main())
