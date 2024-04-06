import numpy as np

from dataclasses import dataclass
from src.solution import MachineSolution, Solution
from src.utils import build_transaction_solution
from typing import Iterable, Optional


@dataclass
class TransactionScheduleTask:
    """Single Transaction Scheduling Task

    Attributes:
        id: Identifier of the problem.
        num_machines: Number of available machines for the transactions.
        lengths: Length of each transaction.
        conflicts: Conflicts betweeen transactions that prevent two
            transactions to run at the same time.
        solution: Order of one optimal solution. In order to interpret this,
            please see the provided material describing the data.
    """
    id: int
    num_machines: int
    lengths: np.ndarray
    conflicts: np.ndarray
    solution: np.ndarray

    @property
    def num_transactions(self) -> int:
        """Getter for the number of transactions.

        Returns:
            Number of transactions.
        """
        return len(self.lengths)

    @property
    def optimal_solution(self) -> Solution:
        """Getter for the representation of the optimal Solution.

        In order to reduce calculation costs, the optimal solution is lazy
        loaded.

        Returns:
            Representation of the optimal Solution.
        """
        if hasattr(self, '_optimal_solution'):
            return self._optimal_solution

        solution = self.solution_order_to_solution(
            solution_order=self.solution)
        self._optimal_solution = solution
        return self._optimal_solution

    def estimate_execution_time(
            self,
            lengths: Optional[np.ndarray] = None
            ) -> float:
        """Estimate the execution length.

        The estimation is calculated based R in paper [1].

        Args:
            lengths: The lengths to use for this problem. Defaults to
                `self.lengths`.

        Returns:
            Estimated execution length.
        """
        if lengths is None:
            lengths = self.lengths

        medium_execution_time = lengths.sum() / self.num_machines
        max_transaction_length = lengths.max()
        execution_time = max(medium_execution_time, max_transaction_length)
        return execution_time

    def solution_order_to_solution(
            self,
            solution_order: Iterable[int]
            ) -> Solution:
        """Build a solution for this task from a solution order.

        Args:
            solution_order: Order of the transactions in the processing time
                format.

        Returns:
            Solution: Solution generated from the given order.
        """
        machines = [MachineSolution([]) for _ in range(self.num_machines)]

        for transaction_id in solution_order:
            # Find machine with lowest processing time.
            machine = min(machines, key=lambda x: x.processing_time)
            duration = self.lengths[transaction_id]

            build_transaction_solution(
                machine=machine,
                machines=machines,
                transaction_id=transaction_id,
                duration=duration,
                conflicts=self.conflicts
            )

        return Solution(machines=machines)
