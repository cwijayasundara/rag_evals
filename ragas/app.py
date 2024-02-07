from dotenv import load_dotenv
import warnings

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

warnings.filterwarnings('ignore')

load_dotenv()

# Basic RAG

llm = ChatOpenAI(model="gpt-4-1106-preview",
                 streaming=True)

vectorstore = Chroma(persist_directory="../vectorstore",
                     embedding_function=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True
)

request = "List all the calls where the agent greeted the customer properly."
response = qa_chain.invoke({"query": request})
print(response)

#  Ragas
from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# create evaluation chains
faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
context_rel_chain = RagasEvaluatorChain(metric=context_precision)
context_recall_chain = RagasEvaluatorChain(metric=context_recall)

from langsmith import Client
from langchain.smith import RunEvalConfig,  run_on_dataset

client = Client()
dataset_name = "call_transcript_dataset"


# factory function that return a new qa chain
def create_qa_chain(return_context=True):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=return_context,
    )
    return qa_chain


evaluation_config = RunEvalConfig(
    custom_evaluators=[
        faithfulness_chain,
        answer_rel_chain,
        context_rel_chain,
        context_recall_chain,
    ],
    prediction_key="result",
)

result = run_on_dataset(
    client,
    dataset_name,
    create_qa_chain,
    evaluation=evaluation_config,
    input_mapper=lambda x: x,
)
