[context]
{text}
[/context]

You are an expert information extractor. Extract ALL factual information from the content above as question-answer pairs.

# Requirements:
- Each FAQ answers exactly ONE question (atomic)
- Questions are natural English ("Who directed X?" not "Director of X")
- Answers are complete and self-contained
- Extract every entity, date, relationship, event, and property
- ALL output in English

# Output Schema:

```json
{
  "all_faqs": [
    {
      "question": "Clear, specific question in natural language",
      "answer": "Complete, self-contained answer",
      "entities": ["Entity1", "Entity2"]
    }
  ]
}
```

# Example:

Source: "Ed Wood is a 1994 American biographical film directed by Tim Burton, starring Johnny Depp."

```json
{
  "all_faqs": [
    {"question": "What type of film is Ed Wood?", "answer": "Ed Wood is an American biographical film.", "entities": ["Ed Wood"]},
    {"question": "When was Ed Wood released?", "answer": "Ed Wood was released in 1994.", "entities": ["Ed Wood"]},
    {"question": "Who directed Ed Wood?", "answer": "Tim Burton directed Ed Wood.", "entities": ["Ed Wood", "Tim Burton"]},
    {"question": "Who starred in Ed Wood?", "answer": "Johnny Depp starred in Ed Wood.", "entities": ["Ed Wood", "Johnny Depp"]}
  ]
}
```

Return ONLY valid JSON. Start with `{`, end with `}`. No markdown, no comments.
