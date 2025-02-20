

Skey = "sk-AR1B3wFOs6I2rRfVr7z1ZyVtVDcdwNzeqcg5NTIaD6fPC4Eu"
class NEREvaluator:
    """Named Entity Recognition evaluation pipeline.

    Attributes:
        llm: Language model instance
        output_buffer: Results output buffer
        metrics: Evaluation metrics storage
    """

    def __init__(self, llm_model: str):
        self.llm = ChatOllama(model=llm_model)
        self.output_buffer = io.StringIO()
        self.metrics = {
            'total_entities': 0,
            'correct_entities': 0,
            'wrong_class': 0,
            'missed_entities': 0,
            'partial_matches': 0,
            'head_mismatches': 0
        }

    def _log(self, message: str, console=True):
        """Log messages to buffer and console."""
        print(message, file=self.output_buffer)
        if console:
            print(message)

    def _clean_response(self, response: str) -> list:
        """Clean and split LLM response."""
        if not response or any(kw in response for kw in ["This ", "There ", " no "]):
            return []

        return [item.strip()
                for item in response.replace('"', '')
                    .replace("'", "")
                    .replace("\n", "")
                    .replace("*", "")
                    .split(',')]

    def evaluate_sentence(self, row_data: tuple):
        """Process single sentence evaluation."""
        sentence, entities, classes = row_data
        self._log(f"\nProcessing sentence: {sentence[:50]}...")

        # Run LLM inference
        loc_entities = self._get_loc_entities(sentence)
        misc_entities = self._get_misc_entities(sentence)

        # Process results
        filtered_loc = self._filter_entities(loc_entities, misc_entities)
        self._update_metrics(filtered_loc, entities, classes)

    def _get_loc_entities(self, sentence: str) -> list:
        """Extract LOC entities from sentence."""
        template = """
        You are a NER assistant. Extract LOCATION entities from:
        {sentence}
        Respond in "Entity, Entity" format or empty string.
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | CommaSeparatedListOutputParser()
        return chain.invoke({"sentence": sentence})

    # Additional helper methods follow the same pattern...

    def save_results(self, a: float, b: float, model_name: str):
        """Save evaluation results to file."""
        output = self.output_buffer.getvalue()
        path = RESULT_PATH.format(a=a, b=b, model=model_name)

        with open(path, 'w', encoding=DEFAULT_ENCODING) as f:
            f.write(output)

        self.output_buffer.close()