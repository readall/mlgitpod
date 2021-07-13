import os

from transformers import pipeline

def screen_clear():
   # for mac and linux(here, os.name is 'posix')
   if os.name == 'posix':
      _ = os.system('clear')
   else:
      # for windows platfrom
      _ = os.system('cls')

screen_clear()

# classifier = pipeline("sentiment-analysis")
# print(classifier("I've been waiting for a HuggingFace course my whole life."))
# print(classifier([
#     "I've been waiting for a HuggingFace course my whole life.", 
#     "I hate this so much!"
# ]))


# classifier = pipeline("zero-shot-classification")
# print(classifier(
#     "This is a course about the Transformers library",
#     candidate_labels=["education", "politics", "business"],
# ))


# # from transformers import pipeline

# generator = pipeline("text-generation")
# print(generator("In this course, we will teach you how to"))


# # Using any model from the Hub in a pipeline
# generator = pipeline("text-generation", model="distilgpt2")
# print(generator(
#     "In this course, we will teach you how to",
#     max_length=30,
#     num_return_sequences=2,
# ))

# screen_clear()

# # The next pipeline you’ll try is fill-mask. The idea of this task is to fill in the blanks in a given text:
# unmasker = pipeline("fill-mask")
# print(unmasker("This course will teach you all about <mask> models.", top_k=2))



# # Named entity recognition
# # Named entity recognition (NER) is a task where the model has to find which parts of the input text correspond to entities such as persons, locations, or organizations. Let’s look at an example:
# ner = pipeline("ner", grouped_entities=True)
# print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

# # The question-answering pipeline answers questions using information from a given context:
# question_answerer = pipeline("question-answering")
# print(question_answerer(
#     question="Where do I work?",
#     context="My name is Sylvain and I work at Hugging Face in Brooklyn"
# ))


# # Summarization
# # Summarization is the task of reducing a text into a shorter text while keeping all (or most) of the important aspects # referenced in the text. Here’s an example:

# summarizer = pipeline("summarization")
# print(summarizer("""
#     America has changed dramatically during recent years. Not only has the number of 
#     graduates in traditional engineering disciplines such as mechanical, civil, 
#     electrical, chemical, and aeronautical engineering declined, but in most of 
#     the premier American universities engineering curricula now concentrate on 
#     and encourage largely the study of engineering science. As a result, there 
#     are declining offerings in engineering subjects dealing with infrastructure, 
#     the environment, and related issues, and greater concentration on high 
#     technology subjects, largely supporting increasingly complex scientific 
#     developments. While the latter is important, it should not be at the expense 
#     of more traditional engineering.

#     Rapidly developing economies such as China and India, as well as other 
#     industrial countries in Europe and Asia, continue to encourage and advance 
#     the teaching of engineering. Both China and India, respectively, graduate 
#     six and eight times as many traditional engineers as does the United States. 
#     Other industrial countries at minimum maintain their output, while America 
#     suffers an increasingly serious decline in the number of engineering graduates 
#     and a lack of well-educated engineers.
# """))

# # Translation
# # For translation, you can use a default model if you provide a language pair in the task name (such as "translation_en_to_fr"), but the easiest way is to pick the model you want to use on the Model Hub. Here we’ll try translating from French to English:
# translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
# en_text = translator("Ce cours est produit par Hugging Face.")
# print(en_text, en_text[0], type(en_text))
# # print(translator("Ce cours est produit par Hugging Face."))
# # the translator pipeline returns a list of dictionaries. Each element of list is a dictionary
# translator_en_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
# print(translator_en_hi(en_text[0]['translation_text']))

