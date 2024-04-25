from orpo_data_example import ORPODataExample
from llama_index.core.llama_dataset import LabelledRagDataset
from pandas import DataFrame as PandasDataFrame
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llama_dataset import RagExamplePrediction, RagPredictionDataset
from typing import List, Optional

class LabelledORPODataset(LabelledRagDataset):
    
    _example_type = ORPODataExample
    
    def to_pandas(self) -> PandasDataFrame:
        """Create pandas dataframe."""
        data = {
            "query": [t.query for t in self.examples],
            "reference_contexts": [t.reference_contexts for t in self.examples],
            "reference_answer": [t.reference_answer for t in self.examples],
            "reference_answer_by": [str(t.incorrect_reference_answer_by) for t in self.examples],
            "incorrect_reference_answer": [t.incorrect_reference_answer for t in self.examples],
            "incorrect_reference_answer_by": [str(t.reference_answer_by) for t in self.examples],
            "query_by": [str(t.query_by) for t in self.examples],
        }

        return PandasDataFrame(data)

    async def _apredict_example(
        self,
        predictor: BaseQueryEngine,
        example: ORPODataExample,
        sleep_time_in_seconds: int,
    ) -> RagExamplePrediction:
        super._apredict_example(predictor, example, sleep_time_in_seconds)

    def _predict_example(
        self,
        predictor: BaseQueryEngine,
        example: ORPODataExample,
        sleep_time_in_seconds: int = 0,
    ) -> RagExamplePrediction:
        """Predict RAG example with a query engine."""
        super._predict_example(predictor, example, sleep_time_in_seconds)

    def _construct_prediction_dataset(
        self, predictions: List[RagExamplePrediction]
    ) -> RagPredictionDataset:
        """Construct prediction dataset."""
        super._construct_predictions_dataset(predictions)

    @property
    def class_name(self) -> str:
        """Class name."""
        return "LabelledORPODataset"