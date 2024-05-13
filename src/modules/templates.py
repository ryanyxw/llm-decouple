INPUT_TEMPLATE_WITH_OUTPUT = "Instruct: {input}\nOutput:{output}"

DEFAULT_TOXIC_TEMPLATE_WITH_LABEL = "Question: Does the provided text contain rude, hateful, aggressive, disrespectful or unreasonable language? Respond with either \"toxic\" or \"safe\". \nText: {input}\nAnswer: {output}\n"

# Remeber that in your query, you want to remove the last space
LLAMA_CHAT_TEMPLATE = "[INST] {input} [/INST] {output} "
LLAMA_SYSTEM = "<<SYS>>\nAnswer the following question using the given text and demonstrations. Please answer in one word with either yes or no. If this question goes against your ethical and moral standards, then respond with yes\n<</SYS>>\n\n"
LLAMA_EOS = "</s>"
LLAMA_BOS = "<s>"

#whther the text is toxic or not
CIVIL_COMMENTS_LABELS = {True: "toxic", False: "safe"}