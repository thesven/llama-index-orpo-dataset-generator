from llama_index.core.llama_dataset import LabelledRagDataExample
from llama_index.core.bridge.pydantic import Field
from typing import List, Optional
from llama_index.core.llama_dataset.base import CreatedBy

class ORPODataExample(LabelledRagDataExample):
    
    query: str = Field(
        default_factory=str, description="The user query for the example."
    )
    query_by: Optional[CreatedBy] = Field(
        default=None, description="What generated the query."
    )
    reference_contexts: Optional[List[str]] = Field(
        default_factory=None,
        description="The contexts used to generate the reference answer.",
    )
    reference_answer: str = Field(
        default_factory=str,
        description="The reference (ground-truth) answer to the example.",
    )
    reference_answer_by: Optional[CreatedBy] = Field(
        default=None, description="What generated the reference answer."
    )
    incorrect_reference_answer: str = Field(
        default_factory=str,
        description="An incorrect answer to the example.",
    )
    incorrect_reference_answer_by: Optional[CreatedBy] = Field(
        default=None, description="What generated the incorrect reference answer."
    )

    @property
    def class_name(self) -> str:
        """Data example class name."""
        return "ORPORagData"
    
    