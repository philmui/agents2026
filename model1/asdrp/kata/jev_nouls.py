from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # take environment variables from .env.

from langchain_typesafe import Noul, TypeSafeClassifier

classifier = TypeSafeClassifier()

response = classifier.invoke({
    "state": (
        "The deploy failed twice and customers are seeing 500s. "
        "CAn someone look now?"
    ),
    "questions": {
        "urgent": Noul(
            instructions="Is this urgent?"),
    },
})

urgent = response.nouls["urgent"].noul

print(f"Is this urgent? {urgent}")
