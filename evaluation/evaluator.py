
from .metrics import RagasMetric
from .data import Query


class MetricsEvaluator:
    def __init__(self, metrics: list[RagasMetric], attempts: int = 1):
        self.metrics = metrics
        self.attempts = attempts

    async def _score_with_retry(self, metric, query_obj: Query):
        last_exception = None

        for _ in range(self.attempts):
            try:
                return await metric.score(
                    query=query_obj.query,
                    response=query_obj.response,
                    contexts=query_obj.contexts,
                    reference=query_obj.reference,
                )
            except Exception as e:
                last_exception = e

        return {"exception": str(last_exception)}

    async def evaluate(self, query_obj: Query):
        results = {}

        for metric in self.metrics:
            value = await self._score_with_retry(metric, query_obj)
            results[metric.name] = value

        return results
