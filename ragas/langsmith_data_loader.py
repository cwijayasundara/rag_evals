from dotenv import load_dotenv

load_dotenv()

from langsmith import Client
from langsmith.utils import LangSmithError

client = Client()
dataset_name = "call_transcript_dataset"

call_transcript_1 = """call_id: 271354 call_date_time: 08:07:2023 14:22:05 call_duration: 00:05:32 agent: Lucas Green 
call_transcript: agent(Lucas Green) How can I be of service today? customer(Sarah Bennett) I'm having trouble with a 
payment not going through. agent(Lucas Green) Let's resolve that for you, Sarah. Can I have the last four digits of 
your SSN for security purposes? customer(Sarah Bennett) Sure, they are 4321. agent(Lucas Green) Great, we'll check on 
that payment now.
"""

call_transcript_2 = """call_id: 324869 call_date_time: 10:06:2023 11:15:43 call_duration: 00:05:32 agent: Elizabeth 
Jenkins call_transcript: agent(Elizabeth Jenkins) Hi there! How can I help you today? customer(Aaron Mitchell) Hi, 
I need to report a lost card. agent(Elizabeth Jenkins) I'm sorry to hear that, Aaron. For security, can I have your 
email address and telephone number? customer(Aaron Mitchell) Certainly, my email is a.mitchell@email.com and my phone 
number is (555) 123-4567. agent(Elizabeth Jenkins) Thank you. I'll block your card immediately and issue a new one.
"""

eval_questions = ["list the call with call_id is 271354",
                  "list the call where agent is Lucas Green",
                  "list all the calls that happened on 10:06:2023"]

eval_answers = [call_transcript_1, call_transcript_1, call_transcript_2]


examples = [
    {"query": q, "ground_truths": [eval_answers[i]]}
    for i, q in enumerate(eval_questions)
]

try:
    # check if dataset exists
    dataset = client.read_dataset(dataset_name=dataset_name)
    print("using existing dataset: ", dataset.name)
except LangSmithError:
    # if not create a new one with the generated query examples
    dataset = client.create_dataset(
        dataset_name=dataset_name, description="call_transcript_dataset"
    )
    for e in examples:
        client.create_example(
            inputs={"query": e["query"]},
            outputs={"ground_truths": e["ground_truths"]},
            dataset_id=dataset.id,
        )

    print("Created a new dataset: ", dataset.name)
