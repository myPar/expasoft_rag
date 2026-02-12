from ragas import SingleTurnSample


class RagasMetric:
    def __init__(self, metric_cls, llm, name: str, **metric_kwargs):
        self.name = name
        self.metric = metric_cls(llm=llm, **metric_kwargs)

    async def score(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: list[str] | None = None,
        reference: str | None = None,
    ):
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=contexts,
            reference=reference,
        )

        return await self.metric.single_turn_ascore(sample)
