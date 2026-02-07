"""Prompt templates for MLE-STAR agents."""

from typing import Dict, Any, List, Optional
import json


class PlannerPrompts:
    """Prompt templates for the Planner agent."""

    @staticmethod
    def task_decomposition(
        task_description: str,
        task_type: str,
        constraints: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None,
        iteration: int = 0,
        num_strategies: int = 3
    ) -> str:
        """
        Generate prompt for task decomposition into strategies.

        Args:
            task_description: Description of the task
            task_type: Type of task (classification, regression, etc.)
            constraints: Task constraints
            dataset_info: Optional dataset profiling information
            iteration: Current iteration number
            num_strategies: Number of strategies to generate

        Returns:
            Formatted prompt string
        """
        # Build prompt parts
        parts = [
            f"You are an expert ML engineering planner. Your task is to decompose the following ML problem into {num_strategies} different solution strategies.",
            "",
            "**Task Description:**",
            task_description,
            "",
            f"**Task Type:** {task_type}",
            "",
            "**Constraints:**",
            json.dumps(constraints, indent=2),
        ]

        # Add dataset information if available
        if dataset_info and 'data_profile' in dataset_info:
            profile = dataset_info['data_profile']
            if profile:
                parts.extend([
                    "",
                    "**Dataset Information:**",
                    f"Shape: {profile.get('num_rows', 'Unknown')} rows × {profile.get('num_cols', 'Unknown')} columns",
                ])

                # Add feature details
                if 'columns' in profile and profile['columns']:
                    parts.append("")
                    parts.append("**Available Features:**")
                    for col, info in profile['columns'].items():
                        dtype = info.get('dtype', 'unknown')
                        unique = info.get('unique_values', 0)
                        missing = info.get('missing_pct', 0)
                        parts.append(
                            f"- {col}: {dtype} ({unique} unique, {missing:.1f}% missing)"
                        )

                # Add sample data preview
                if 'sample_data' in profile:
                    try:
                        sample_df = profile['sample_data']
                        parts.extend([
                            "",
                            "**Sample Data (first 3 rows):**",
                            str(sample_df.head(3).to_string()),
                        ])
                    except Exception:
                        pass

        parts.extend([
            "",
            f"**Iteration:** {iteration}",
            "",
            f"Please generate {num_strategies} distinct solution strategies. For each strategy, provide:"
        ])

        prompt = "\n".join(parts) + """
1. A clear approach description
2. Subtasks breakdown (specific, actionable steps)
3. Required libraries/tools
4. Expected complexity
5. Success criteria

Return your response as a JSON array of strategies with this structure:
```json
[
  {{
    "id": "strategy_1",
    "name": "Strategy Name",
    "approach": "High-level description of the approach",
    "subtasks": [
      {{
        "id": "subtask_1",
        "description": "Specific task to perform",
        "dependencies": [],
        "estimated_time": 60
      }}
    ],
    "requirements": ["pandas", "scikit-learn"],
    "complexity": "medium",
    "success_criteria": ["Accuracy > 0.9", "Code runs without errors"]
  }}
]
```

Focus on generating diverse, practical strategies that explore different approaches to solving the problem.
"""
        return prompt

    @staticmethod
    def strategy_refinement(
        task_description: str,
        previous_results: List[Dict[str, Any]],
        feedback: Dict[str, Any]
    ) -> str:
        """
        Generate prompt for refining strategies based on feedback.

        Args:
            task_description: Original task description
            previous_results: Results from previous iterations
            feedback: Feedback from verification

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert ML engineering planner. Based on previous attempts and feedback, generate improved solution strategies.

**Original Task:**
{task_description}

**Previous Results:**
{json.dumps(previous_results, indent=2)}

**Feedback:**
{json.dumps(feedback, indent=2)}

Analyze what worked and what didn't in the previous attempts. Generate 3 refined strategies that:
1. Address the issues identified in the feedback
2. Build on successful elements from previous attempts
3. Explore alternative approaches for failed attempts

Return your response in the same JSON format as before.
"""
        return prompt


class ExecutorPrompts:
    """Prompt templates for the Executor agent."""

    @staticmethod
    def code_generation(
        subtasks: List[Dict[str, Any]],
        requirements: List[str],
        context: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate prompt for code implementation.

        Args:
            subtasks: List of subtasks to implement
            requirements: Required libraries
            context: Additional context (data paths, etc.)
            dataset_info: Optional dataset profiling information

        Returns:
            Formatted prompt string
        """
        parts = [
            "You are an expert Python developer specializing in ML engineering. Implement a complete, working solution for the following subtasks.",
            "",
            "**Subtasks to Implement:**",
            json.dumps(subtasks, indent=2),
            "",
            "**Required Libraries:**",
            ', '.join(requirements) if requirements else 'No specific requirements',
            "",
            "**Context:**",
            json.dumps(context, indent=2) if context else '{}',
        ]

        # Add dataset information if available
        if dataset_info and 'data_profile' in dataset_info:
            profile = dataset_info['data_profile']
            if profile and 'columns' in profile:
                parts.extend([
                    "",
                    "**Dataset Schema:**",
                ])
                for col, info in profile['columns'].items():
                    dtype = info.get('dtype', 'unknown')
                    missing = info.get('missing_pct', 0)
                    parts.append(f"- {col}: {dtype} ({missing:.1f}% missing)")

                # Add target candidates
                if 'target_candidates' in profile and profile['target_candidates']:
                    parts.extend([
                        "",
                        f"**Likely Target Column(s):** {', '.join(profile['target_candidates'])}"
                    ])

        parts.extend([
            "",
            "**Requirements:**"
        ])

        prompt = "\n".join(parts) + """
1. Write complete, executable Python code
2. Include all necessary imports
3. Add error handling
4. Include comments for complex logic
5. Make code modular and clean
6. Save any outputs or models as specified

**Code Structure:**
```python
# Imports
import ...

# Configuration
...

# Main implementation
def main():
    # Implement subtasks here
    ...

if __name__ == "__main__":
    main()
```

Return ONLY the complete Python code wrapped in ```python code blocks. Do not include explanations outside the code.
"""
        return prompt

    @staticmethod
    def code_fixing(
        original_code: str,
        error_message: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate prompt for fixing code errors.

        Args:
            original_code: Code that produced an error
            error_message: Error message
            context: Additional context

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert Python debugger. Fix the following code that produced an error.

**Original Code:**
```python
{original_code}
```

**Error Message:**
```
{error_message}
```

**Context:**
{json.dumps(context, indent=2)}

Analyze the error and provide a corrected version of the code. Ensure:
1. The root cause of the error is addressed
2. Similar errors are prevented
3. The code still accomplishes the original goal
4. All edge cases are handled

Return ONLY the corrected Python code wrapped in ```python code blocks.
"""
        return prompt


class VerifierPrompts:
    """Prompt templates for the Verifier agent."""

    @staticmethod
    def verification(
        strategy: Dict[str, Any],
        execution_result: Dict[str, Any],
        success_criteria: List[str]
    ) -> str:
        """
        Generate prompt for result verification.

        Args:
            strategy: The strategy that was executed
            execution_result: Results from code execution
            success_criteria: Criteria for success

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert code reviewer and ML validator. Verify the execution results against the strategy and success criteria.

**Strategy:**
{json.dumps(strategy, indent=2)}

**Execution Result:**
```
Status: {execution_result.get('status', 'unknown')}
Output: {execution_result.get('output', 'N/A')[:1000]}
Error: {execution_result.get('error', 'None')}
```

**Success Criteria:**
{json.dumps(success_criteria, indent=2)}

**Verification Tasks:**
1. Check if the code executed successfully
2. Verify that all subtasks were completed
3. Evaluate results against success criteria
4. Identify any bugs, errors, or issues
5. Assess code quality and best practices
6. Provide specific, actionable feedback

Return your verification as a JSON object with this structure:
```json
{{
  "status": "success" | "partial" | "failed",
  "score": 0.0-1.0,
  "criteria_met": {{
    "criterion_1": true,
    "criterion_2": false
  }},
  "issues": [
    {{
      "severity": "critical" | "major" | "minor",
      "description": "Issue description",
      "location": "Where the issue is"
    }}
  ],
  "feedback": {{
    "strengths": ["What worked well"],
    "weaknesses": ["What needs improvement"],
    "suggestions": ["Specific improvement suggestions"]
  }},
  "metrics": {{
    "execution_time": 10.5,
    "memory_used": "200MB",
    "custom_metrics": {{}}
  }}
}}
```

Be thorough and specific in your verification.
"""
        return prompt

    @staticmethod
    def code_quality_check(
        code: str,
        language: str = "python"
    ) -> str:
        """
        Generate prompt for code quality assessment.

        Args:
            code: Code to assess
            language: Programming language

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a senior code reviewer. Assess the quality of the following {language} code.

**Code to Review:**
```{language}
{code}
```

**Assessment Criteria:**
1. Correctness: Does the code work as intended?
2. Readability: Is the code clear and well-structured?
3. Efficiency: Are there performance issues?
4. Best Practices: Does it follow {language} best practices?
5. Error Handling: Are errors handled appropriately?
6. Security: Are there security vulnerabilities?

Return your assessment as a JSON object:
```json
{{
  "overall_score": 0.0-1.0,
  "correctness": 0.0-1.0,
  "readability": 0.0-1.0,
  "efficiency": 0.0-1.0,
  "best_practices": 0.0-1.0,
  "error_handling": 0.0-1.0,
  "security": 0.0-1.0,
  "issues": [
    {{
      "category": "correctness",
      "severity": "high",
      "line": 42,
      "description": "Issue description",
      "suggestion": "How to fix"
    }}
  ],
  "summary": "Overall assessment summary"
}}
```
"""
        return prompt


class PromptFormatter:
    """Utility class for formatting prompts."""

    @staticmethod
    def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from LLM response.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON object or None if not found
        """
        import re

        # Try to find JSON code block
        json_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass

        # Try to parse entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def extract_code_from_response(response: str, language: str = "python") -> Optional[str]:
        """
        Extract code from LLM response.

        Args:
            response: LLM response text
            language: Programming language

        Returns:
            Extracted code or None if not found
        """
        import re

        # Try to find code block
        code_pattern = rf'```{language}\s*(.*?)\s*```'
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try without language specifier
        code_pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        return None

    @staticmethod
    def format_list(items: List[str], bullet: str = "-") -> str:
        """
        Format list as bullet points.

        Args:
            items: List of items
            bullet: Bullet character

        Returns:
            Formatted string
        """
        return "\n".join(f"{bullet} {item}" for item in items)

    @staticmethod
    def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
        """
        Truncate text to maximum length.

        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        return text[:max_length - len(suffix)] + suffix
