
from .metrics import RagasMetric
from .data import Query


def get_safe_score(x):
    return 0.0 if x is None or x != x else x  # NaN != NaN


class MetricsEvaluator:
    def __init__(self, metrics: list[RagasMetric], attempts: int = 1):
        self.metrics = metrics
        self.attempts = attempts

    async def _score_with_retry(self, metric, query: Query, response):
        last_exception = None

        for _ in range(self.attempts):
            try:
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

    async def evaluate(self, query: Query, query_engine):
        results = {"metrics": {}}
        # put query to rag:
        try:
            response = await query_engine.aquery(query.text)
        except Exception as e:
            return {"exception": str(e)}
        
        # calc result metrics:
        for metric in self.metrics:
            value = await self._score_with_retry(metric, query, response)
            if isinstance(value, dict) and "exception" in value:
                return {"exception": f"exception at metric - {metric}: " + value['exception']}
            results['metrics'][metric.name] = get_safe_score(value)
        
        # add context to result:
        selected_chunks = response.source_nodes
        context = [chunk.text for chunk in selected_chunks]

        results["query"] = query.text
        results["response"] = str(response)
        results["context"] = context
        results["expected_answer"] = query.answer_text

        return results
