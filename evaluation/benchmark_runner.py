import asyncio
import json
import os
from tqdm.asyncio import tqdm_asyncio
from .data import Query
from typing import List


class AsyncBenchmarkRunner:
    def __init__(self, evaluator, max_concurrent: int = 8):
        self.evaluator = evaluator
        self.sem = asyncio.Semaphore(max_concurrent)    # semaphore to limit max concurrent requests

    async def run(
        self,
        bench_data: List[Query],
        save_results = True,
        dst_dir = ".",
        bench_name = "demo",
    ):
        results = {"scores": None, "experiments": []}

        async def worker(query):
            async with self.sem:
                return query.uid, await self.evaluator.evaluate(query)

        tasks = [
            asyncio.create_task(worker(q)) for q in bench_data
        ]

        metric_names = [m.name for m in self.evaluator.metrics]
        sums = {name: 0.0 for name in metric_names}
        ok = 0

        # do parallel requests:
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            uid, result = await coro

            if not isinstance(result, dict) or "exception" in result:   # bad result or exception inside
                results["experiments"].append(
                    {"uid": uid, "result": result}
                )
                continue
            
            ok += 1

            for name in metric_names:
                value = result.get(name)

                # ignore metric-level exception
                if isinstance(value, dict) and "exception" in value:
                    continue

                if value is not None:
                    sums[name] += float(value)

            results["experiments"].append(
                {"uid": uid, "result": result}
            )

        # avoid division by zero
        if ok > 0:
            results["scores"] = {
                name: sums[name] / ok for name in metric_names
            }
        else:
            results["scores"] = {
                name: 0.0 for name in metric_names
            }

        # save bench results:
        if save_results:
            path = os.path.join(dst_dir, f"{bench_name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return results
