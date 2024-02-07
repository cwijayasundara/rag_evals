from dotenv import load_dotenv
import warnings
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
load_dotenv()

warnings.filterwarnings('ignore')

# load the Wikipedia page and create index
loader = WebBaseLoader("https://en.wikipedia.org/wiki/New_York_City")
index = VectorstoreIndexCreator().from_loaders([loader])

# create the QA chain
llm = ChatOpenAI()

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=index.vectorstore.as_retriever(),
    return_source_documents=True
)

question = "How did New York City get its name?"
result = qa_chain({"query": question})
print(result)

# test dataset
eval_questions = [
    "What is the population of New York City as of 2020?",
    "Which borough of New York City has the highest population?",
    "What is the economic significance of New York City?",
    "How did New York City get its name?",
    "What is the significance of the Statue of Liberty in New York City?",
]

eval_answers = [
    "8,804,190",
    "Brooklyn",
    "New York City's economic significance is vast, as it serves as the global financial capital, housing Wall Street "
    "and major financial institutions. Its diverse economy spans technology, media, healthcare, education, and more, "
    "making it resilient to economic fluctuations. NYC is a hub for international business, attracting global "
    "companies, and boasts a large, skilled labor force. Its real estate market, tourism, cultural industries, "
    "and educational institutions further fuel its economic prowess. The city's transportation network and global "
    "influence amplify its impact on the world stage, solidifying its status as a vital economic player and cultural "
    "epicenter.",
    "New York City got its name when it came under British control in 1664. King Charles II of England granted the "
    "lands to his brother, the Duke of York, who named the city New York in his own honor.",
    "The Statue of Liberty in New York City holds great significance as a symbol of the United States and its ideals "
    "of liberty and peace. It greeted millions of immigrants who arrived in the U.S. by ship in the late 19th and "
    "early 20th centuries, representing hope and freedom for those seeking a better life. It has since become an "
    "iconic landmark and a global symbol of cultural diversity and freedom.",
]

examples = [
    {"query": q, "ground_truths": [eval_answers[i]]}
    for i, q in enumerate(eval_questions)
]

result = qa_chain({"query": eval_questions[1]})
print(result)

# Add Ragas for evaluation

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

predictions = qa_chain.batch(examples)

print("evaluating faithfulness_chain...")
r = faithfulness_chain.evaluate(examples, predictions)
print(r)

print("evaluating answer_rel_chain...")
r = answer_rel_chain.evaluate(examples, predictions)
print(r)

print("evaluating context_rel_chain...")
r = context_rel_chain.evaluate(examples, predictions)
print(r)

print("evaluating context_recall_chain...")
r = context_recall_chain.evaluate(examples, predictions)
print(r)

# Add LangSmith

from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import Client

client = Client()
dataset_name = "NYC test"


# factory function that return a new qa chain
def create_qa_chain(return_context=True):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=index.vectorstore.as_retriever(),
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

