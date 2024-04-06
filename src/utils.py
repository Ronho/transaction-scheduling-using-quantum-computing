import numpy as np

from src.solution import MachineSolution, TransactionSolution
from typing import List


def build_transaction_solution(
        machine: MachineSolution,
        machines: List[MachineSolution],
        transaction_id: int,
        duration: float,
        conflicts: np.ndarray
        ) -> None:
    """Creates a new TransactionSolution and adds it to the machine.

    We ensure that no conflicting transactions run at the same time and that no
    two transactions run at the same time on the same machine.

    Args:
        machine: Machine which executes the transaction.
        machines: All machines.
        transaction_id: Id of the transaction to execute.
        duration: Execution time of the transaction.
        conflicts: 2D-Array containing the conflicts between transactions.
    """
    if len(machine.transactions) > 0:
        earliest_possible = machine.transactions[-1].end_time
    else:
        earliest_possible = 0

    # For every other machine, check that no other conflicting
    # transaction is running after earliest_possible, otherwise postpone
    # execution of the current transaction.
    for other_machine in machines:
        if id(other_machine) != id(machine):
            for omt in other_machine.transactions:
                # Check that a conflict exist and whether their time
                # windows are overlapping.
                if conflicts[omt.id][transaction_id] > 0 \
                        and omt.start_time <= earliest_possible + duration \
                        and omt.end_time >= earliest_possible:
                    earliest_possible = max(
                        omt.end_time,
                        earliest_possible
                    )

    machine.transactions.append(TransactionSolution(
        id=transaction_id,
        start_time=earliest_possible,
        end_time=earliest_possible + duration,
        duration=duration
    ))
