from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from trulens_eval import Tru
from util.trulence_eval_util import get_tru_recorder
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

vectorstore = Chroma(persist_directory="vectorstore",
                     embedding_function=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-4-1106-preview",
                   streaming=True)

output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = (setup_and_retrieval
         | prompt
         | model
         | output_parser)

request = "List all the calls where the agent greeted the customer properly."
response = chain.invoke(request)
print(response)

# load the eval dataset
eval_questions = []
with open('trulense/eval_questions.txt', 'r') as file:
    for line in file:
        item = line.strip()
        eval_questions.append(item)

print(len(eval_questions))
print(eval_questions)

#  add trulense evals

tru = Tru()
tru.reset_database()

app_id = "call_eval_engine"

tru_recorder = get_tru_recorder(chain, app_id=app_id)

with tru_recorder as recording:
    for question in eval_questions:
        response = chain.invoke(question)

records, feedback = tru.get_records_and_feedback(app_ids=[app_id])

print(feedback)
print(records.head())

tru.run_dashboard()
