"""Verifier agent implementation."""

from typing import Dict, Any
from ..core.base_agent import BaseAgent, AgentConfig, AgentRole
from ..core.message import Message, MessageType
from ..utils.prompt_templates import VerifierPrompts, PromptFormatter


class VerifierAgent(BaseAgent):
    """
    Verifier Agent - Uses Qwen2.5-Coder-14B locally.

    Responsibilities:
    - Validate execution results against success criteria
    - Check code quality and correctness
    - Detect bugs and issues
    - Provide structured feedback
    - Score solutions (0-1 scale)

    Example:
        >>> verifier = VerifierAgent(config, local_model)
        >>> response = await verifier.handle_message(verification_request)
    """

    def __init__(self, config: AgentConfig, model_interface):
        """
        Initialize Verifier agent.

        Args:
            config: Agent configuration
            model_interface: Local model (Qwen2.5-Coder-14B)
        """
        super().__init__(config, model_interface)
        self.prompts = VerifierPrompts()
        self.formatter = PromptFormatter()

        self.logger.info("Verifier agent initialized")

    async def process(self, message: Message) -> Message:
        """
        Process verification request.

        Args:
            message: Verification request message

        Returns:
            Verification result message

        Raises:
            ValueError: If message type not supported
        """
        if message.msg_type != MessageType.VERIFICATION_REQUEST:
            raise ValueError(
                f"Verifier only supports VERIFICATION_REQUEST, got {message.msg_type}"
            )

        strategy = message.content['strategy']
        execution_result = message.content['execution_result']

        self.logger.info(f"Verifying strategy: {strategy.get('name', 'Unknown')}")

        # Verify result
        verification = await self._verify_result(strategy, execution_result)

        return message.create_reply(
            msg_type=MessageType.VERIFICATION_RESULT,
            content=verification
        )

    async def _verify_result(
        self,
        strategy: Dict[str, Any],
        execution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify execution result against criteria.

        Args:
            strategy: Strategy that was executed
            execution: Execution result

        Returns:
            Verification dictionary with score and feedback
        """
        self.logger.debug("Verifying execution result")

        # Build verification prompt
        prompt = self.prompts.verification(
            strategy=strategy,
            execution_result=execution,
            success_criteria=strategy.get('success_criteria', [])
        )

        try:
            # Generate verification
            response = await self._generate_response(
                prompt,
                temperature=0.1,  # Very low for consistent evaluation
                max_tokens=2000
            )

            # Parse verification result
            verification = self._parse_verification(response)

            # Add objective metrics
            verification['objective_metrics'] = self._compute_objective_metrics(
                execution
            )

            # Compute final score if not present
            if 'score' not in verification:
                verification['score'] = self._compute_score(verification)

            self.logger.info(
                f"Verification complete: "
                f"status={verification.get('status')}, "
                f"score={verification.get('score', 0):.2f}"
            )

            return verification

        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            # Return fallback verification
            return {
                'status': 'error',
                'score': 0.0,
                'error': str(e),
                'objective_metrics': self._compute_objective_metrics(execution)
            }

    def _parse_verification(self, response: str) -> Dict[str, Any]:
        """
        Parse verification from LLM response.

        Args:
            response: LLM response text

        Returns:
            Verification dictionary
        """
        # Try to extract JSON from response
        verification = self.formatter.extract_json_from_response(response)

        if verification is None:
            # Fallback: create verification from text
            self.logger.warning("Could not extract JSON, using fallback")
            verification = {
                'status': 'partial',
                'score': 0.5,
                'feedback': {
                    'summary': response[:500]
                }
            }

        # Ensure required fields
        verification.setdefault('status', 'partial')
        verification.setdefault('score', 0.5)
        verification.setdefault('criteria_met', {})
        verification.setdefault('issues', [])
        verification.setdefault('feedback', {})
        verification.setdefault('metrics', {})

        # Ensure feedback structure
        feedback = verification['feedback']
        feedback.setdefault('strengths', [])
        feedback.setdefault('weaknesses', [])
        feedback.setdefault('suggestions', [])

        return verification

    def _compute_objective_metrics(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute objective metrics from execution result.

        Args:
            execution: Execution result

        Returns:
            Dictionary of objective metrics
        """
        metrics = {
            'execution_success': execution.get('status') == 'success',
            'execution_time': execution.get('execution_time', 0.0),
            'has_output': bool(execution.get('output')),
            'has_errors': bool(execution.get('error')),
            'exit_code': execution.get('exit_code', -1)
        }

        return metrics

    def _compute_score(self, verification: Dict[str, Any]) -> float:
        """
        Compute overall score from verification data.

        Args:
            verification: Verification dictionary

        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0

        # Base score on status
        status = verification.get('status', 'failed')
        if status == 'success':
            score = 1.0
        elif status == 'partial':
            score = 0.5
        else:
            score = 0.0

        # Adjust based on objective metrics
        obj_metrics = verification.get('objective_metrics', {})
        if not obj_metrics.get('execution_success', False):
            score *= 0.5

        # Adjust based on criteria met
        criteria_met = verification.get('criteria_met', {})
        if criteria_met:
            met_ratio = sum(1 for v in criteria_met.values() if v) / len(criteria_met)
            score = (score + met_ratio) / 2

        # Adjust based on issues
        issues = verification.get('issues', [])
        if issues:
            critical_issues = sum(1 for i in issues if i.get('severity') == 'critical')
            if critical_issues > 0:
                score *= 0.5

        # Ensure score is in valid range
        return max(0.0, min(1.0, score))

    async def assess_code_quality(self, code: str) -> Dict[str, Any]:
        """
        Assess code quality independently.

        Args:
            code: Code to assess

        Returns:
            Quality assessment dictionary
        """
        self.logger.debug("Assessing code quality")

        prompt = self.prompts.code_quality_check(code)

        try:
            response = await self._generate_response(
                prompt,
                temperature=0.1,
                max_tokens=1500
            )

            assessment = self.formatter.extract_json_from_response(response)

            if assessment is None:
                assessment = {
                    'overall_score': 0.5,
                    'summary': response[:500]
                }

            return assessment

        except Exception as e:
            self.logger.error(f"Code quality assessment failed: {e}")
            return {
                'overall_score': 0.0,
                'error': str(e)
            }

    def validate_input(self, message: Message) -> bool:
        """
        Validate input message.

        Args:
            message: Message to validate

        Returns:
            True if valid, False otherwise
        """
        if message.msg_type != MessageType.VERIFICATION_REQUEST:
            return False

        required = ['strategy', 'execution_result']
        return all(k in message.content for k in required)

    def validate_output(self, response: Message) -> bool:
        """
        Validate output message.

        Args:
            response: Response message

        Returns:
            True if valid, False otherwise
        """
        if response.msg_type == MessageType.VERIFICATION_RESULT:
            return 'score' in response.content

        return response.msg_type == MessageType.ERROR
