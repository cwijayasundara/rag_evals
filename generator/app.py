from dotenv import load_dotenv
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.openai import (
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from pathlib import Path
import os

load_dotenv()


class CallDetails(BaseModel):
    call_id: str
    call_date_time: str
    call_duration: str
    agent: str
    call_transcript: str


# sample data

examples = [
    {
        "example": """call_id: 123456, call_date_time: 01:02:2024 11:05:45, call_duration: 00:05:00, agent: Dan 
        Smith, call_transcript: agent(dan smith) hello mr johns good morning how are you today customer(tom johns) I 
        am well thanks this is regarding one of my credit card transactions agent(dan smith) sure sir please could 
        you tell me the last 4 digits of your ssn and your email address please customer(tom johns) sure my ssn is 
        1234 and my email address is tom johns at gmail.com agent(dan smith) thank you sir how can I help you today 
        customer(tom johns) I have a transaction on my credit card that I do not recognize agent(dan smith) sure sir 
        please could you tell me the date and amount of the transaction customer(tom johns) sure the date was 
        01:02:2024 and the amount was 100 dollars agent(dan smith) thank you sir I can see the transaction on your 
        account customer(tom johns) yes I do not recognize it agent(dan smith) sure sir I will raise a dispute for 
        this transaction and you will receive a confirmation email within 24 hours customer(tom johns) thank you 
        agent(dan smith) is there anything else I can help you with today customer(tom johns) no thank you agent(dan 
        smith) thank you for calling have a nice day customer(tom johns) thank you agent(dan smith) goodbye customer(
        tom johns) goodbye"""
    },
    {
        "example": """call_id: 234567, call_date_time: 02:12:2023 00:30:45, call_duration: 00:05:10, agent: joe jobs, 
        call_transcript: agent(joe jobs) how can I help you today? customer(tom hanks) I have a transaction on my 
        current account that I did not make agent(joe jobs) can you tell me your account number please? customer(tom 
        hanks) its 34-33456-99 agent(joe jobs) thank you can you tell me the date and amount of the transaction 
        customer(tom hanks) today and the amount was 100 dollars agent(joe jobs) I will raise a dispute for this 
        transaction and you will receive a confirmation email within 24 hours customer(tom hanks) thank you agent(joe 
        jobs) welcome"""
    }
]

#  Craft a Prompt Template

OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)

#  Create a Data Generator
synthetic_data_generator = create_openai_data_generator(
    output_schema=CallDetails,
    llm=ChatOpenAI(
        temperature=1, model="gpt-4-1106-preview"
    ),
    prompt=prompt_template,
)

extra_txt = """ You are an expert in contact centre operations for a bank.
Please follow the below guidelines when generating the synthetic contact centre data 
- Make sure the 
agent greets the customer properly as given in the example in some samples and omit this in some other samples 
- 
call_id, call_date_time,call_duration,agent,call_transcript and customer name must be chosen at  random 
- Make sure 
to include the agent checking for at least 2 items from the below security details 
- Email address - Last 4 digits of 
the SSN - Telephone number 
- Omit the above security checks in some of the samples."""

#  Generate Data
synthetic_results = synthetic_data_generator.generate(
    subject="call_details",
    extra=extra_txt,
    runs=10,
)

"""loop through the records and save the content of each record with the header in a file called call_id.txt inside a 
directory called data. if the directory does not exist, create it."""

directory = 'data'

Path(directory).mkdir(parents=True, exist_ok=True)

for record in synthetic_results:
    call_id = record.call_id
    filename = os.path.join(directory, f'{call_id}.txt')
    with open(filename, 'w') as file:
        file.write(f"call_id: {record.call_id}\n")
        file.write(f"call_date_time: {record.call_date_time}\n")
        file.write(f"call_duration: {record.call_duration}\n")
        file.write(f"agent: {record.agent}\n")
        file.write(f"call_transcript: {record.call_transcript}\n")
        file.close()
