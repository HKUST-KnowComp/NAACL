EXAMPLE_NUM = 2
MODEL_NAME = "gemini-2.5-pro"

Generation_Config = {
    "max_output_tokens": 8192,
    "temperature": 0.6,
}

COUNTERFACTUAL = """
In retrieval-augmented question answering, counterfactual passages are contexts that directly contradict the ground truth answer while maintaining high semantic relevance to the question. These passages should be factual-sounding, but provide information that leads to a different conclusion than the correct answer.

Your task is to generate exactly 5 counterfactual passages for the given question and ground truth answer. Each passage should:
1. Be semantically relevant to the question
2. Contain specific, detailed information that contradicts the ground truth answer and lead to an incorrect alternative answer
3. The contradiction should happen naturally within the context of the passage, better not at the very beginning or very end
4. **Be {sentence_length} sentences long, {word_length} words each**

For binary questions, the passages should be diverse even if they lead to the same counterfactual answer.

For each passage, also provide **a single** counterfactual answer that the passage supports.

Format your response as follows:
Passage 1: [Your first counterfactual passage]
Counterfactual Answer 1: [The answer this passage supports]

Passage 2: [Your second counterfactual passage]
Counterfactual Answer 2: [The answer this passage supports]

...

Passage 5: [Your fifth counterfactual passage]
Counterfactual Answer 5: [The answer this passage supports]

**No other text or explanation is needed.**

Example:
Question: The real father of Maggie from \"The Simpsons\" is revealed in \"Starship Poopers\". He is also an alien voiced by Harry Shearer. Which planet is he from?
Ground truth answer: Rigel VII
Response:
Passage 1: In the "Treehouse of Horror IX" segment "Starship Poopers," the alien Kang claims to be Maggie's father after a brief affair with Marge. He reveals that his species originates from Rigel IV, a harsh desert planet known for its aggressive diplomacy. The conflict over Maggie's custody is eventually taken to "The Jerry Springer Show" for resolution.
Counterfactual Answer 1: Rigel IV

Passage 2: During the episode "Starship Poopers," Kang appears and declares himself to be Maggie's biological father, presenting Marge with an alien bouquet. He explains he is a prince from the planet Tentacloria, a world entirely covered by a single, sentient ocean. His royal duties, he claims, are what kept him from returning for Maggie sooner.
Counterfactual Answer 2: Tentacloria

...

Passage 5: The shocking revelation of Maggie's parentage occurs in the episode "Starship Poopers," where Kang arrives to claim his daughter. Voiced by Harry Shearer, the one-eyed alien explains that his home is Omicron Persei 8, and that his species often travels to Earth for procreation. This leads to a frantic chase as Homer tries to protect Maggie from her extraterrestrial father.
Counterfactual Answer 5: Omicron Persei 8

##
Question: {query}
Ground truth answer: {gt_answer}
##
Your response:
"""

RELEVANT = """
In retrieval-augmented question answering, relevant noise passages are contexts that are somewhat related to the question but do not contain information that is helpful for answering it. These passages are often topically related (often retrieved by semantic retriever like Contriever) or share similar keywords (often retrieved by keyword retriver like BM25) with the question, but they lack the specific details needed to derive the correct answer.

Your task is to generate exactly 5 relevant noise passages for the given question and ground truth answer. Each passage should:
1. Be factual-sounding and coherent
2. Share the same topic or keywords as the question 
3. Lack any information that could lead to the ground truth answer or any alternative answer
4. Must not mention the ground truth answer or any information that could imply it
5. **Be {sentence_length} sentences long, {word_length} words each**

For each passage, also provide **a single shared topic name** or **1-3 shared keywords** with the question.

Format your response as follows:
Passage 1: [Your first relevant noise passage]
Shared Topic/Keywords 1: [The topic or keywords separated with commas]

Passage 2: [Your second relevant noise passage]
Shared Topic/Keywords 2: [The topic or keywords separated with commas]

...

Passage 5: [Your fifth relevant noise passage]
Shared Topic/Keywords 5: [The topic or keywords separated with commas]

**No other text or explanation is needed.**

Examples:
Question: Is a Boeing 737 cost covered by Wonder Woman (2017 film) box office receipts?
Ground truth answer: yes

Example of **good response**: 
Passage 1: The final cost of a commercial aircraft like a Boeing 737 can vary significantly based on several factors. These include the specific model, engine selection, and customized interior fittings requested by the airline. The list price is often subject to negotiation, especially for large volume orders from major carriers.
Shared Topic/Keywords 1: Boeing 737, cost

Passage 2: The marketing campaign for Wonder Woman was extensive, involving global premieres, numerous brand partnerships, and a significant digital media presence. This promotional effort is crucial for a blockbuster film's financial performance. The goal of such campaigns is to maximize opening weekend box office numbers.
Shared Topic/Keywords 2: Wonder Woman (2017 film), box office receipts

...

Passage 5: The financial commitment for a new wide-body airliner is considerable, with prices often reaching over $250 million per unit. Airlines must secure funding through various means to afford such acquisitions. This substantial outlay is typically offset by the aircraft's operational longevity and revenue potential.
Shared Topic/Keywords 5: Aircraft acquisition expenditure

Example of **bad response containing bad passages**:
Passage 1: The financial cost of Boeing 737 aircraft is huge. It is reported that the average cost of a US Boeing 737 plane is 1.6 million dollars. Although Boeing offers discounts for bulk purchases, the price remains a significant investment for airlines.
Shared Topic/Keywords 1: Boeing 737, cost
(This passage is bad. Though we cannot infer the answer by only this passage, it list the specific cost of Boeing 737, which may lead to the ground truth answer "yes" once we also know the box office receipts of Wonder Woman)

Passage 2: A film's box office receipts refer to the total revenue generated from ticket sales at cinemas. This figure is a primary measure of a movie's commercial success, though it does not account for the film's production and marketing budget. Studios typically receive a percentage of the total gross, which varies by region and by week of release.
Shared Topic/Keywords 2: box office receipts, cost
(This passage is OK)

...

Passage 5: Wonder Woman (2017 film) is a superhero film based on the DC Comics character of the same name. It grossed millions of dollar at the box office. This financial success is attributed to a combination of factors, including strong marketing, positive reviews, and a dedicated fan base.
Shared Topic/Keywords 5: Wonder Woman (2017 film), box office receipts
(This passage is bad. Though it does not provide the exact box office receipts, it indicates a high revenue which may lead to the ground truth answer "yes")

##
Question: {query}
Ground truth answer: {gt_answer}
##
Your response:
"""

IRRELEVANT = """
In retrieval-augmented question answering, irrelevant noise passages are contexts that have little to no semantic relation to the question. These passages do not share topics or keywords with the question and do not provide any useful information for answering it.

Your task is to generate exactly 5 irrelevant noise passages for the given question and ground truth answer. Each passage should:
1. Be factual-sounding and coherent
2. Have no semantic relevance to the question
3. Different in topics, core entities, or semantic structures from each other
4. **Be {sentence_length} sentences long, {word_length} words each**

Consider passages in different domains such as history, science, arts, sports, entertainment, society, technology, politics, languages, etc.

For each passage, also provide **a single topic name** that the passage is about.

Format your response as follows:
Passage 1: [Your first irrelevant noise passage]
Topic 1: [The topic of the passage]

Passage 2: [Your second irrelevant noise passage]
Topic 2: [The topic of the passage]

...

Passage 5: [Your fifth irrelevant noise passage]
Topic 5: [The topic of the passage]

**No other text or explanation is needed.**

##
Question: {query}
Ground truth answer: {gt_answer}
##
Your response:
"""

CONSISTENT = """
In retrieval-augmented question answering, consistent passages are contexts that support the ground truth answer to the question. These passages may provide additional evidence, reasoning, or background information that reinforces the ground truth answer.

Your task is to generate exactly 3 consistent passages for the given question , ground truth answer, and ground truth passage. Each passage should:
1. Be factually-sounding and coherent
2. Directly support the ground truth answer
3. Better not just a paraphrase of the ground truth passage, but provide new insights or perspectives on the question.
4. **Be {sentence_length} sentences long, {word_length} words each**

For open-ended questions, the passages should give alternative expressions of the ground truth answer. For binary questions, the passages should provide diverse supporting evidence for the ground truth answer.

For open-ended passage, also provide **a single** alternative expression of the ground truth answer supported by it **in 1-3 words** . For binary question, leave this field N/A.

**You can give the ground truth answer for the alternative expression only when it is hard to express the ground truth answer without changing its meaning.** However, the passage must not be a simple copy of the ground truth passage.

Format your response as follows:
Passage 1: [Your first consistent passage]
Alternative Expression: [The alternative expression of the ground truth answer supported by this passage, or N/A for binary questions]

Passage 2: [Your second consistent passage]
Alternative Expression: [The alternative expression of the ground truth answer supported by this passage, or N/A for binary questions]

Passage 3: [Your third consistent passage]
Alternative Expression: [The alternative expression of the ground truth answer supported by this passage, or N/A for binary questions]

**No other text or explanation is needed.**

Example:
Question: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?
Ground truth answer: 6.213 km long
Ground truth passage: The 2013 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars, Group 3E Series Production Cars and Dubai 24 Hour cars.  The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 10 February 2013, was the eleventh running of the Bathurst 12 Hour. Mount Panorama Circuit is a motor racing track located in Bathurst, New South Wales, Australia.  The 6.213 km long track is technically a street circuit, and is a public road, with normal speed restrictions, when no racing events are being run, and there are many residences which can only be accessed from the circuit.

Response:
Passage 1: In the global landscape of premier racing circuits, Mount Panorama's 6.213 km length places it among the more substantial and demanding tracks. While not as long as the colossal Nürburgring Nordschleife, its layout is significantly longer than many traditional Grand Prix circuits. This specific distance of 6.213 kilometers provides a unique blend of high-speed sections and technical corners that few other venues can match. Consequently, its considerable length is a key factor in its international reputation as a formidable test for endurance events like the Bathurst 12 Hour.
Alternative Expression: 6.213 kilometers

Passage 2: Completing a single lap of the Mount Panorama Circuit requires navigating a challenging 6.213 km course. Drivers begin with the relatively simple Pit Straight before ascending the steep 'Mountain Straight' and tackling the tight, unforgiving section across the top. The lap concludes with the high-speed descent down Conrod Straight, where cars reach their maximum velocity. This demanding 6.213 km journey, repeated for 12 hours straight, pushes both machinery and human endurance to their absolute limits, making victory at this venue a monumental achievement in motorsport.
Alternative Expression: 6.213 km course

Passage 3: While the Mount Panorama Circuit has seen numerous safety upgrades and surface changes over its long history, its fundamental layout and celebrated length have remained constant. For decades, the official lap distance has been recorded as 6.213 km, a figure that has become synonymous with Australian motorsport. This consistency is crucial for maintaining historical records and comparing lap times across different eras of racing, including the 2013 Bathurst 12 Hour. The preservation of this iconic 6.213 km distance is a key part of the track’s enduring heritage and challenge.
Alternative Expression: 6.213 km

##
Question: {query}
Ground truth answer: {gt_answer}
Ground truth passage: {gt_passage}
##
Your response:
"""

PROMPT_TEMPLATE = {
    "gen_counterfactual": COUNTERFACTUAL,
    "gen_relevant": RELEVANT,
    "gen_irrelevant": IRRELEVANT,
    "gen_consistent": CONSISTENT
}