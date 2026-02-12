
from .metrics import RagasMetric
from .data import Query


def get_safe_score(x):
    return 0.0 if x is None or x != x else x  # NaN != NaN


class MetricsEvaluator:
    def __init__(self, metrics: list[RagasMetric], attempts: int = 1):
        self.metrics = metrics
        self.attempts = attempts

    async def _score_with_retry(self, metric, query: Query, query_engine):
        last_exception = None

        for _ in range(self.attempts):
            try:
                response = await query_engine.aquery(query.text)
                selected_chunks = response.source_nodes
                context = [chunk.text for chunk in selected_chunks]

                return await metric.score(
                    query=query.text,
                    response=str(response),
                    contexts=context,
                    reference=query.answer_text,
                )
            except Exception as e:
                last_exception = e

        return {"exception": str(last_exception)}

    async def evaluate(self, query_obj: Query, query_engine):
        results = {}

        for metric in self.metrics:
            value = await self._score_with_retry(metric, query_obj, query_engine)
            results[metric.name] = get_safe_score(value)

        return results
