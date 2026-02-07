"""STAR (Search, Test, And Refine) workflow orchestrator."""

from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio
import logging

from .message import Message, MessageType
from .state_manager import StateManager


class WorkflowPhase(Enum):
    """Phases in the STAR workflow."""

    SEARCH = "search"
    EVALUATION = "evaluation"
    REFINEMENT = "refinement"
    COMPLETED = "completed"
    FAILED = "failed"


class STARWorkflow:
    """
    STAR (Search, Test, And Refine) workflow orchestrator.

    This class implements the MLE-STAR methodology:
    1. **Search**: Generate multiple solution strategies
    2. **Evaluation**: Execute and verify each strategy
    3. **Refinement**: Improve based on feedback or select best

    The workflow iterates until a solution meets success criteria
    or max iterations are reached.

    Example:
        >>> workflow = STARWorkflow(planner, executor, verifier, state_manager)
        >>> result = await workflow.execute(task)
    """

    def __init__(
        self,
        planner,
        executor,
        verifier,
        state_manager: StateManager,
        max_iterations: int = 5,
        parallel_strategies: int = 3
    ):
        """
        Initialize STAR workflow.

        Args:
            planner: Planner agent instance
            executor: Executor agent instance
            verifier: Verifier agent instance
            state_manager: State manager for persistence
            max_iterations: Maximum number of refinement iterations
            parallel_strategies: Number of strategies to explore per iteration
        """
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.state_manager = state_manager
        self.max_iterations = max_iterations
        self.parallel_strategies = parallel_strategies

        self.logger = logging.getLogger("mle_star.workflow")
        self.logger.info(
            f"STAR workflow initialized: "
            f"max_iterations={max_iterations}, "
            f"parallel_strategies={parallel_strategies}"
        )

    async def execute(self, task: Any) -> Dict[str, Any]:
        """
        Execute STAR workflow for a task.

        Args:
            task: Task object to solve

        Returns:
            Result dictionary with status and outcome

        Example:
            >>> result = await workflow.execute(task)
            >>> if result['status'] == 'success':
            ...     print("Task completed!")
        """
        # Initialize task state
        task_id = self.state_manager.create_task(task)
        self.logger.info(f"Starting STAR workflow for task {task_id}")

        try:
            # Iterative STAR loop
            for iteration in range(self.max_iterations):
                self.logger.info(
                    f"Iteration {iteration + 1}/{self.max_iterations}"
                )

                # Phase 1: SEARCH - Generate solution strategies
                self.logger.info("Phase 1: Search - Generating strategies")
                strategies = await self._search_phase(task, iteration)

                if not strategies:
                    self.logger.warning("No strategies generated, aborting")
                    return self._create_failure_result("No strategies generated")

                # Phase 2: EVALUATION - Execute and verify strategies
                self.logger.info(
                    f"Phase 2: Evaluation - Testing {len(strategies)} strategies"
                )
                results = await self._evaluation_phase(strategies)

                # Record iteration
                self.state_manager.update_iteration(task_id, {
                    'iteration': iteration,
                    'strategies_count': len(strategies),
                    'results': results
                })

                # Phase 3: REFINEMENT - Analyze and decide
                self.logger.info("Phase 3: Refinement - Analyzing results")
                decision = await self._refinement_phase(results, task)

                # Check if task is completed
                if decision['action'] == 'complete':
                    self.logger.info("Task completed successfully!")
                    final_result = {
                        'status': 'success',
                        'result': decision['best_result'],
                        'iterations': iteration + 1,
                        'task_id': task_id
                    }
                    self.state_manager.complete_task(task_id, final_result)
                    return final_result

                elif decision['action'] == 'refine':
                    self.logger.info("Refining approach for next iteration")
                    # Continue to next iteration with feedback
                    continue

                else:  # 'fail'
                    self.logger.warning("Task failed")
                    return self._create_failure_result(
                        decision.get('reason', 'Unknown failure')
                    )

            # Max iterations reached
            self.logger.warning(
                f"Max iterations ({self.max_iterations}) reached without success"
            )

            # Select best result from all iterations
            best_result = self._select_best_result(task_id)

            final_result = {
                'status': 'partial_success',
                'result': best_result,
                'iterations': self.max_iterations,
                'reason': 'max_iterations_reached',
                'task_id': task_id
            }

            self.state_manager.complete_task(task_id, final_result)
            return final_result

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            self.state_manager.fail_task(task_id, str(e))
            return self._create_failure_result(str(e))

    async def _search_phase(
        self,
        task: Any,
        iteration: int
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple solution strategies (SEARCH phase).

        Args:
            task: Task object
            iteration: Current iteration number

        Returns:
            List of strategy dictionaries
        """
        # Create task request message
        msg = Message(
            msg_type=MessageType.TASK_REQUEST,
            sender="workflow",
            receiver="planner",
            content={
                'task': task.to_dict(),
                'iteration': iteration,
                'num_strategies': self.parallel_strategies
            }
        )

        # Get strategies from planner
        response = await self.planner.handle_message(msg)

        if response.msg_type == MessageType.ERROR:
            self.logger.error(
                f"Planner error: {response.content.get('error')}"
            )
            return []

        strategies = response.content.get('strategies', [])
        self.logger.info(f"Generated {len(strategies)} strategies")

        return strategies

    async def _evaluation_phase(
        self,
        strategies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute and verify each strategy (EVALUATION phase).

        Args:
            strategies: List of strategies to evaluate

        Returns:
            List of result dictionaries (one per strategy)
        """
        results = []

        for i, strategy in enumerate(strategies):
            self.logger.info(
                f"Evaluating strategy {i+1}/{len(strategies)}: "
                f"{strategy.get('name', 'Unknown')}"
            )

            try:
                # Execute strategy
                exec_msg = Message(
                    msg_type=MessageType.EXECUTION_REQUEST,
                    sender="workflow",
                    receiver="executor",
                    content={'strategy': strategy}
                )

                exec_response = await self.executor.handle_message(exec_msg)

                if exec_response.msg_type == MessageType.ERROR:
                    self.logger.warning(
                        f"Execution error for strategy {i+1}: "
                        f"{exec_response.content.get('error')}"
                    )
                    results.append({
                        'strategy': strategy,
                        'execution': {'status': 'error', 'error': exec_response.content.get('error')},
                        'verification': {'score': 0.0, 'status': 'failed'}
                    })
                    continue

                # Verify result
                verify_msg = Message(
                    msg_type=MessageType.VERIFICATION_REQUEST,
                    sender="workflow",
                    receiver="verifier",
                    content={
                        'strategy': strategy,
                        'execution_result': exec_response.content.get('result', {})
                    }
                )

                verify_response = await self.verifier.handle_message(verify_msg)

                if verify_response.msg_type == MessageType.ERROR:
                    self.logger.warning(
                        f"Verification error for strategy {i+1}: "
                        f"{verify_response.content.get('error')}"
                    )
                    results.append({
                        'strategy': strategy,
                        'execution': exec_response.content.get('result', {}),
                        'verification': {'score': 0.0, 'status': 'error'}
                    })
                    continue

                # Combine execution and verification results
                result = {
                    'strategy': strategy,
                    'execution': exec_response.content.get('result', {}),
                    'verification': verify_response.content,
                    'code': exec_response.content.get('code', '')
                }

                results.append(result)

                self.logger.info(
                    f"Strategy {i+1} score: "
                    f"{verify_response.content.get('score', 0):.2f}"
                )

            except Exception as e:
                self.logger.error(
                    f"Error evaluating strategy {i+1}: {e}",
                    exc_info=True
                )
                results.append({
                    'strategy': strategy,
                    'execution': {'status': 'error', 'error': str(e)},
                    'verification': {'score': 0.0, 'status': 'error'}
                })

        return results

    async def _refinement_phase(
        self,
        results: List[Dict[str, Any]],
        task: Any
    ) -> Dict[str, str]:
        """
        Analyze results and decide next action (REFINEMENT phase).

        Args:
            results: Results from evaluation phase
            task: Original task

        Returns:
            Decision dictionary with 'action' and additional info
        """
        # Sort results by score
        results_sorted = sorted(
            results,
            key=lambda x: x['verification'].get('score', 0),
            reverse=True
        )

        best = results_sorted[0] if results_sorted else None

        if not best:
            return {'action': 'fail', 'reason': 'No results available'}

        best_score = best['verification'].get('score', 0)

        self.logger.info(f"Best score this iteration: {best_score:.2f}")

        # Success threshold
        success_threshold = 0.8

        if best_score >= success_threshold:
            # Task completed successfully
            return {
                'action': 'complete',
                'best_result': best,
                'score': best_score
            }

        # Check if we have any partial success
        if best_score >= 0.5:
            # Refine - we have something to work with
            return {
                'action': 'refine',
                'best_result': best,
                'score': best_score,
                'feedback': best['verification'].get('feedback', {})
            }

        # Low scores - might need complete rethink
        if best_score < 0.3:
            return {
                'action': 'refine',
                'best_result': best,
                'score': best_score,
                'major_refinement': True
            }

        # Default: continue refining
        return {
            'action': 'refine',
            'best_result': best,
            'score': best_score
        }

    def _select_best_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Select best result across all iterations.

        Args:
            task_id: Task ID

        Returns:
            Best result dictionary or None
        """
        iterations = self.state_manager.get_iterations(task_id)

        best_result = None
        best_score = 0.0

        for iteration_data in iterations:
            for result in iteration_data.get('results', []):
                score = result.get('verification', {}).get('score', 0)
                if score > best_score:
                    best_score = score
                    best_result = result

        return best_result

    def _create_failure_result(self, reason: str) -> Dict[str, Any]:
        """
        Create failure result dictionary.

        Args:
            reason: Failure reason

        Returns:
            Failure result dictionary
        """
        return {
            'status': 'failed',
            'reason': reason,
            'result': None
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get workflow statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'max_iterations': self.max_iterations,
            'parallel_strategies': self.parallel_strategies,
            'planner_stats': self.planner.get_stats(),
            'executor_stats': self.executor.get_stats(),
            'verifier_stats': self.verifier.get_stats()
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"STARWorkflow("
            f"max_iterations={self.max_iterations}, "
            f"parallel_strategies={self.parallel_strategies}"
            f")"
        )
