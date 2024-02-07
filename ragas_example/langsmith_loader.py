from dotenv import load_dotenv

load_dotenv()

from langsmith import Client
from langsmith.utils import LangSmithError

client = Client()
dataset_name = "NYC test"

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

try:
    # check if dataset exists
    dataset = client.read_dataset(dataset_name=dataset_name)
    print("using existing dataset: ", dataset.name)
except LangSmithError:
    # if not create a new one with the generated query examples
    dataset = client.create_dataset(
        dataset_name=dataset_name, description="NYC test dataset"
    )
    for e in examples:
        client.create_example(
            inputs={"query": e["query"]},
            outputs={"ground_truths": e["ground_truths"]},
            dataset_id=dataset.id,
        )

    print("Created a new dataset: ", dataset.name)

