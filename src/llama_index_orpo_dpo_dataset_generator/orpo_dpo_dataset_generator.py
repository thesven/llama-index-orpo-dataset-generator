from __future__ import annotations

import asyncio
import re
from typing import List, Optional

from llama_index_orpo_dpo_dataset_generator.orpo_data_example import ORPODataExample
from llama_index_orpo_dpo_dataset_generator.labelled_orpo_dataset import LabelledORPODataset

from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core import Document, ServiceContext, SummaryIndex
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.ingestion import run_transformations
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
    LabelledRagDataExample,
    LabelledRagDataset,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.node import KeywordNodePostprocessor
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    TransformComponent,
)
from llama_index.core.settings import (
    Settings,
    llm_from_settings_or_context,
    transformations_from_settings_or_context,
)

class ORPODPODatasetGenerator(RagDatasetGenerator):
    
    """Generate dataset (question/ question-answer-incorrect answer sets) \
    based on the given documents.

    NOTE: this is a beta feature, subject to change!

    Args:
        nodes (List[Node]): List of nodes. (Optional)
        service_context (ServiceContext): Service Context.
        num_questions_per_chunk: number of question to be \
        generated per chunk. Each document is chunked of size 512 words.
        text_question_template: Question generation template.
        question_gen_query: Question generation query.

    """

    def __init__(
        self,
        nodes: List[BaseNode],
        llm: Optional[LLM] = None,
        num_questions_per_chunk: int = 3,
        text_question_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        question_gen_query: Optional[str] = None,
        metadata_mode: MetadataMode = MetadataMode.NONE,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
        service_context: Optional[ServiceContext] = None,
        
    ) -> None:
        """Init params."""
        super().__init__(
            nodes, llm, num_questions_per_chunk, text_question_template,
            text_qa_template, question_gen_query, metadata_mode,
            show_progress, workers, service_context
        )

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        llm: Optional[LLM] = None,
        transformations: Optional[List[TransformComponent]] = None,
        num_questions_per_chunk: int = 3,
        text_question_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        question_gen_query: Optional[str] = None,
        required_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
        service_context: Optional[ServiceContext] = None,
        incorrect_text_qa_template: BasePromptTemplate = None,
        incorrect_answer_prompt: str = "Given the context answser this question incorrectly. Do not say that it is incorrect. Do not explain why it is in correct. Answer as if you are telling the truth."
    ) -> 'ORPODPODatasetGenerator':
        """Generate dataset from documents."""
        cls.incorrect_text_qa_template = PromptTemplate(incorrect_text_qa_template)
        cls.incorrect_answer_prompt = incorrect_answer_prompt
        return super().from_documents(
            documents, llm, transformations, num_questions_per_chunk,
            text_question_template, text_qa_template, question_gen_query,
            required_keywords, exclude_keywords, show_progress, workers,
            service_context
        )

    async def _agenerate_dataset(
        self,
        nodes: List[BaseNode],
        labelled: bool = False,
    ) -> LabelledRagDataset:
        """Node question generator."""
        query_tasks = []
        examples: List[LabelledRagDataExample] = []
        summary_indices: List[SummaryIndex] = []
        for node in nodes:
            index = SummaryIndex.from_documents(
                [
                    Document(
                        text=node.get_content(metadata_mode=self._metadata_mode),
                        metadata=node.metadata,
                        excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                        excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                        relationships=node.relationships,
                    )
                ],
            )

            query_engine = index.as_query_engine(
                llm=self._llm,
                text_qa_template=self.text_question_template,
                use_async=True,
            )
            task = query_engine.aquery(
                self.question_gen_query,
            )
            query_tasks.append(task)
            summary_indices.append(index)

        responses = await run_jobs(query_tasks, self._show_progress, self._workers)
        for idx, response in enumerate(responses):
            result = str(response).strip().split("\n")
            cleaned_questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            ]
            cleaned_questions = [
                question for question in cleaned_questions if len(question) > 0
            ]
            index = summary_indices[idx]
            reference_context = nodes[idx].text
            model_name = self._llm.metadata.model_name
            created_by = CreatedBy(type=CreatedByType.AI, model_name=model_name)
            if labelled:
                qr_tasks = []
                wrong_answer_tasks = []

                for query in cleaned_questions:
                    print(str(query))
                    # Correct Answer Query
                    qa_query_engine = index.as_query_engine(
                        llm=self._llm,
                        text_qa_template=self.text_qa_template,
                    )
                    qr_task = qa_query_engine.aquery(query)
                    qr_tasks.append(qr_task)

                    # Incorrect Answer Query
                    wrong_qa_query_engine = index.as_query_engine(
                        llm=self._llm,
                        text_qa_template=self.incorrect_text_qa_template,
                    )
                    wrong_qr_task = wrong_qa_query_engine.aquery(f"""
                    {self.incorrect_answer_prompt}
                    
                    Query:
                    {query}
                    """)
                    wrong_answer_tasks.append(wrong_qr_task)

                correct_answers = await run_jobs(qr_tasks, self._show_progress, self._workers)
                incorrect_answers = await run_jobs(wrong_answer_tasks, self._show_progress, self._workers)
                
                for question, correct_resp, incorrect_resp in zip(cleaned_questions, correct_answers, incorrect_answers):
                    example = ORPODataExample(
                        query=question,
                        reference_answer=str(correct_resp),
                        incorrect_reference_answer=str(incorrect_resp),
                        query_by=created_by,
                        reference_answer_by=created_by,
                        incorrect_reference_answer_by=created_by,
                        reference_contexts=[reference_context],
                    )
                    
                    examples.append(example)
            else:
                for query in cleaned_questions:
                    example = LabelledRagDataExample(
                        query=query,
                        reference_answer="",
                        reference_contexts=[reference_context],
                        reference_answer_by=None,
                        query_by=created_by,
                    )
                    examples.append(example)

        # split train/test
        return LabelledORPODataset(examples=examples)

    async def agenerate_questions_from_nodes(self) -> LabelledRagDataset:
        """Generates questions but not the reference answers."""
        return await self._agenerate_dataset(self.nodes, labelled=False)

    async def agenerate_dataset_from_nodes(self) -> LabelledRagDataset:
        """Generates questions for each document."""
        return await self._agenerate_dataset(self.nodes, labelled=True)

    def generate_questions_from_nodes(self) -> LabelledRagDataset:
        """Generates questions but not the reference answers."""
        return asyncio.run(self.agenerate_questions_from_nodes())

    def generate_dataset_from_nodes(self) -> LabelledRagDataset:
        """Generates questions for each document."""
        return asyncio.run(self.agenerate_dataset_from_nodes())

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "text_question_template": self.text_question_template,
            "text_qa_template": self.text_qa_template,
        }

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        super(prompts)
