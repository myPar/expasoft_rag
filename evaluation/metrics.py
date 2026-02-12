from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import AnswerCorrectness
from ragas.metrics import Faithfulness
from ragas.metrics import LLMContextRecall


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
    

def get_main_metrics(judge_llm):
    context_recall = RagasMetric(LLMContextRecall, llm=judge_llm, name='context_recall')
    answer_correctness = RagasMetric(AnswerCorrectness, llm=judge_llm, name='answer_correctness', weights=[1, 0])
    context_precision = RagasMetric(LLMContextPrecisionWithoutReference, llm=judge_llm, name='context_precision')
    faithfulness = RagasMetric(Faithfulness, llm=judge_llm, name='faithfulness')

    return [context_recall, context_precision, answer_correctness, faithfulness]
