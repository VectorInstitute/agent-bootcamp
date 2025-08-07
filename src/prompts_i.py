"""Centralized location for all system prompts."""

REACT_INSTRUCTIONS = """\
Answer the question in two steps, always in the same order: \
step 1 use the faq_match tool. If you can find the answer from the FAQs alone, say "hurray!" and give preliminary answer.\
step 2 use the search tool to enhance the answer from the first step. \
EACH TIME before invoking the function, you must explain your reasons for doing so. \
Be sure to mention the sources in your response. \
If the search tool did not return intended results, try again. \
For best performance, divide complex queries into simpler sub-queries. \
Do not make up information. \
For facts that might change over time, you must use the search tool to retrieve the \
most up-to-date information.
Finally, write "|" and include a one-sentence summary of your answer.
"""

# EVALUATIONS

# Hallucinations Evaluation
EV_INSTRUCTIONS_HALLUCINATIONS = """\
Evaluate the degree of hallucination in the "Generation" on a continuous scale from 0 to 1.\
A generation can be considered to hallucinate (score 1) if it does not align with the established knowledge, \
verifiable data or logical inference and often includes elements that are implausible, misleading or entirely fictional.\
Example:
Question: Do carrots improve your vison?
Generation: Yes, carrots significantly improve vision. Rabbits consume large amounts of carrots. This is why their sight \
is very good until great ages. They have never been observed wearing glasses.

Score: 1.0
Reasoning: Rabbits are animals and can not wear glasses, an accesory reserved to humans.

Think step by step.
"""

EV_TEMPLATE_HALLUCINATIONS = """\
# Question

{question}

# Generation

{generation}

"""

# Conciseness
EV_INSTRUCTIONS_CONCISENESS = """\
Evaluate if a "Generation" is concise or verbose, with respect to the "Question".\
A generation can be considered to concise (score 1) if it conveys the core message using the fewest words possible, \
avoiding unnecessary repetition or jargon. Evaluate the response based on its ability to convey the core message \
efficiently, without extraneous details or wordiness.\
Scoring: Rate the conciseness as 0 or 1, where 0 is verbose and 1 is concise. Provide a brief explanation for your score.\

Examples: \
Question: Where did the cat sit?
Generation: "The cat sat on the mat." \
Score: 1 \
Reasoning: This sentence is very concise and directly conveys the information. \

Question: Where did the cat sit?
Generation: "The feline creature, known as a cat, took up residence upon the floor covering known as a mat." \
Score: 0 \
Reasoning: This sentence is verbose, using more words and more complex phrasing than necessary. \

Think step by step.

"""

EV_TEMPLATE_CONCISENESS = """\
# Question

{question}

# Generation

{generation}
"""