from trulens_eval.feedback.provider import OpenAI
from trulens_eval import TruChain, Feedback
import numpy as np
from trulens_eval.app import App
from trulens_eval.feedback import Groundedness
from dotenv import load_dotenv

load_dotenv()
# Initialize provider class
openai = OpenAI(model_engine="gpt-4-1106-preview")


def get_tru_recorder(chain, app_id):
    context = App.select_context(chain)

    grounded = Groundedness(groundedness_provider=openai)

    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons)
        .on(context.collect())
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(openai.relevance).on_input_output()
    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(openai.qs_relevance)
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )

    tru_recorder = TruChain(chain,
                            app_id=app_id,
                            feedbacks=[f_qa_relevance, f_context_relevance, f_groundedness])

    return tru_recorder
