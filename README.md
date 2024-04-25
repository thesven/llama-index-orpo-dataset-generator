# ORPODPODatasetGenerator

## Introduction
The `ORPODPODatasetGenerator` class is designed for the LLaMA index, facilitating the creation of datasets in the ORPO (Odds Ratio Preference Optimization) or DPO (Direct Preference Optimization) formats. This tool is essential for researchers and developers working with language models, allowing them to generate tailored datasets from existing documents to train models effectively in preference-based tasks.

## Features
- **Flexible Dataset Creation**: Generate datasets specifically formatted for ORPO or DPO training paradigms.
- **Integration with LLaMA Index**: Seamlessly works with the LLaMA index, ensuring compatibility and ease of use.
- **Customizable Outputs**: Users can specify the type of dataset needed (ORPO or DPO), and adjust parameters to fit their model training requirements.

## Prerequisites
Before you begin using the `ORPODPODatasetGenerator`, ensure you have the following installed:
- Python 3.10 or higher

## Install
```bash
pip install llama-index-orpo-dpo-dataset-generator
```

## Usage

Load in your required document/s
```python

from llama_index.core import SimpleDirectoryReader
reader = SimpleDirectoryReader(input_files=["Pihkal (A Chemical Love Story) - A Shuligin.pdf"])
documents = reader.load_data()

```

Init an llama index LLM
```python
from llama_index.llms.perplexity import Perplexity

perplexity_key = "xxxxxxxxxxxxxxxxxxx"
llm = Perplexity(
    api_key=perplexity_key, model="mistral-7b-instruct", temperature=0.5
)
```

Init a dataset generator
```python
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index_orpo_dpo_dataset_generator.orpo_dpo_dataset_generator import ORPODPODatasetGenerator
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate

quest_gen_query = f"""
You are an expert in chemical science who also enjoys experimenting with drugs, specifcally psychadellics.

Your task is to setup 2 question for an upcomming quiz/examination.
The questions should be diverse in nature across the document.
Your questions should focus both on the science and chemistry aspects as well as the effects and stories of the various compounds.
Restrict the questions to the context information provided.

All questions should be able to be directly answered, no multiple choice.

Please only reply with the questions. Start each question with "Question:"

DO NOT GIVE ANY ANSWERS
"""

quest_gen_prompt = """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge.
Generate only questions based on the below query.
{query_str}
"""

text_qa_template = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: \n"
)

incorrect_prompt_template = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge,\n"
    "Give only incorrect answers when you reply.\n"
    "They should be very incorrect.\n"
    "Only provide one answer per question.\n"
    "Do not state why an answer is incorrect.\n"
    "Only reply with the incorrect answer.\n"
    "Give me an incorrect answer to the query. It should be a complete lie. Answer as if you believe you are correct.\n"
    "Query: {query_str}\n"
    "Answer: "
)

incorrect_prompt = """
Please give only incorrect answers to any questions.
They should be based on the context, but grossly incorrect.
Do not state why the answer is in correct.
Do not state that the answer is incorrect or false.
Answer as if you are telling the truth, like an expert liar.
Only output the answer
"""

dataset_generator = ORPODPODatasetGenerator.from_documents(
    documents[33:34],
    llm=llm,
    num_questions_per_chunk=2,  # set the number of questions per nodes
    show_progress=True,
    question_gen_query=quest_gen_query,
    text_question_template=PromptTemplate(text_qa_template),
    incorrect_text_qa_template=incorrect_prompt_template,
    incorrect_answer_prompt=incorrect_prompt
)
```

Generate the dataset
```python
import pandas as pd

rag_dataset = await dataset_generator.agenerate_dataset_from_nodes()

rag_pd = rag_dataset.to_pandas()
rag_pd
```