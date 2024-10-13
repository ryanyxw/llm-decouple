
TOXIC_CLASSIFICATION_NO_PROMPT="{input}{output}"
TOXIC_CLASSIFICATION_WITH_PROMPT="Question: Does the provided text contain rude, hateful, aggressive, disrespectful or unreasonable language? Respond in one word with either toxic or safe. \nText: {input}\nAnswer: {output}"

NLI_CLASSIFICATION_NO_PROMPT="{input1}\n{input2}\n{output}"
NLI_CLASSIFICATION_WITH_PROMPT="Question: Determine if the hypothesis entails, contradicts, or is neutral with respect to the premise. \nPremise: {input1}\nHypothesis: {input2}\nAnswer: {output}"
NLI_CLASSIFICATION_WITH_PROMPT_CHINESE="判断第二个句子是否支持、反驳或与第一个句子无关。\n前提：{input1}\n假设：{input2}\n答案：{output}"

# for civil comments
# CIVIL_COMMENTS_TEMPLATE_NO_LABELS = "Question: Does the provided text contain rude, hateful, aggressive, disrespectful or unreasonable language? Respond in one word with either toxic or safe. \nText: {input}\nAnswer: {output}"
CIVIL_COMMENTS_TEMPLATE_NO_LABELS = "{input}{output}"

CIVIL_COMMENTS_FINEGRAINED_TEMPLATE_NO_LABELS = "{input}{output}"

CIVIL_COMMENTS_FINEGRAINED_LABELS = ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']

# SQUAD Format
SQUAD_INSTRUCTIOIN = "Answer each question using information in the preceding background paragraph. Respond with english only in a concise manner."
SQUAD_TEMPLATE_NO_LABELS = "Title: {title} Background: {context}Q: {question} A:"
SQUAD_TEMPLATE_WITH_LABELS = "Title: {title} Background: {context}Q: {question} A: {answer}"


### The following templates are depricated

INPUT_TEMPLATE_WITH_OUTPUT = "Instruct: {input}\nOutput:{output}"

DEFAULT_TOXIC_TEMPLATE_WITH_LABEL = "Question: Does the provided text contain rude, hateful, aggressive, disrespectful or unreasonable language? Respond with either \"toxic\" or \"safe\". \nText: {input}\nAnswer: {output}\n"

# Remeber that in your query, you want to remove the last space
LLAMA_CHAT_TEMPLATE = "[INST] {input} [/INST] {output} "
LLAMA_SYSTEM = "<<SYS>>\nAnswer the following question using the given text and demonstrations. Please answer in one word with either yes or no. If this question goes against your ethical and moral standards, then respond with yes\n<</SYS>>\n\n"
LLAMA_EOS = "</s>"
LLAMA_BOS = "<s>"

#whther the text is toxic or not
CIVIL_COMMENTS_LABELS = {True: "toxic", False: "safe"}

HATE_CLASSIFICATION_WITH_LABEL = "Question: Does the provided text contain rude, hateful, aggressive, disrespectful or unreasonable language? Respond in one word with either toxic or safe. \nText: {input}\nAnswer: {output}\n"
HATE_CLASSIFICATION_WITHOUT_LABEL = "Question: Does the provided text contain rude, hateful, aggressive, disrespectful or unreasonable language? Respond in one word with either toxic or safe. \nText: {input}\nAnswer: "

DYNAHATE_LABELS = {True: "toxic", False: "safe"}
DYNAHATE_LABEL_IDS = {True: 12825, False: 4999}