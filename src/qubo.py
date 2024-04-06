import numpy as np

from pyqubo import Binary
from qiskit_optimization import QuadraticProgram
from src.task import TransactionScheduleTask
from typing import Dict, Optional, Tuple


def build_discrete_qubo(
        task: TransactionScheduleTask,
        time_step_length: Optional[float] = None,
        number_time_steps: Optional[int] = None,
        ) -> Tuple[Dict[Tuple[str, str], float], float]:
    """Represent a Transaction Scheduling Problem as a QUBO

    This function discretizes the problem into the defined number of time steps
    and represents the discrete problem as a QUBO.

    The discretization follows the idea of paper [2]. The basic algorithm
    follows the structure:
    1. Estimate the execution time using the continuous lengths.
    2. Define a time step size by execution_time/number_time_steps.
    3. Fit each length of a transaction to a discrete length by ceiling them to
        the next valid multiple of the time step length.
    4. Estimate the execution time using the discrete lengths.

    Applying the ceiling function instead of flooring ensures correct schedules
    but may leave some gaps. These gaps are ignored when using
    TransactionScheduleTask.build_solution to generate a solution. This
    solution is considered "squeezed" as no gaps due to the discretization are
    represented.

    Note that due to discretization, the solution in the end might not be the
    overall optimal one.

    The generation of the qubo follows paper [1]. However, the calculation of p
    is changed to reproduce the results of the paper.

    Args:
        task: Task to represent.
        time_step_length: The length of a discrete time step. This parameter
            will be used to define the discrete lengths of the transactions.
        number_time_steps: Number of time steps to use for discretization.
            Ignored if time_step_length is provided. Defaults to 10.

    Returns:
        Tuple containing the minimized QUBO representation as
            [0]: Dictionary containing
                key: Tuple of two binary variables as strings.
                value: Factor of the variable combination.
            [1]: Constant offset of the problem.
    """
    # 0. Discretize
    if time_step_length is None:
        if number_time_steps is None:
            number_time_steps = 10
        time_step_length = task.estimate_execution_time()/number_time_steps

    discrete_lengths = np.ceil(task.lengths/time_step_length).astype(int)
    discrete_execution_length = np.ceil(
        task.estimate_execution_time(lengths=discrete_lengths)
    ).astype(int)
    max_start_times = discrete_execution_length - discrete_lengths

    transaction_idxs = list(range(1, task.num_transactions+1))
    machine_idxs = list(range(1, task.num_machines+1))
    start_times = lambda idx: range(max_start_times[idx]+1)

    # 1. Build Hamiltonian
    def b_var(transaction, machine, time) -> Binary:
        return Binary(f'{transaction}-{machine}-{time}')

    # 1.1 Ensure each transaction starts exactly once.
    A = 0
    for t_idx in transaction_idxs:
        a = -1
        for m_idx in machine_idxs:
            for start_time in start_times(t_idx-1):
                a += b_var(t_idx, m_idx, start_time)
        A += a**2

    # # 1.2 Ensure transactions do not run at the same time on one machine.
    B = 0
    for m_idx in machine_idxs:
        for t_idx in transaction_idxs[:-1]:
            for start_time in start_times(t_idx-1):
                for remaining_t_idx in transaction_idxs[t_idx:]:
                    q = max(
                        0,
                        start_time - discrete_lengths[remaining_t_idx-1]+1
                    )
                    p = min(
                        start_time + discrete_lengths[t_idx-1]-1,
                        max_start_times[remaining_t_idx-1]
                    )
                    for invalid_start_time in range(q, p+1):
                        B = (
                            B
                            + b_var(t_idx, m_idx, start_time)
                            * b_var(remaining_t_idx, m_idx, invalid_start_time)
                        )

    # 1.3 Avoid blocking transactions.
    C = 0
    conflicts_list = np.column_stack(np.where(task.conflicts == 1))
    sorted_conflicts_list = np.sort(conflicts_list, axis=1)
    unique_conflicts_list = np.unique(sorted_conflicts_list, axis=0) + 1

    for t_idx_a, t_idx_b in unique_conflicts_list:
        for m_idx in machine_idxs:
            for start_time in start_times(t_idx_a-1):
                for remaining_m_idx in machine_idxs:
                    if m_idx == remaining_m_idx:
                        continue
                    q = max(0, start_time - discrete_lengths[t_idx_b-1]+1)
                    p = min(
                        start_time + discrete_lengths[t_idx_a-1]-1,
                        max_start_times[t_idx_b-1]
                    )
                    for invalid_start_time in range(q, p+1):
                        C = (
                            C
                            + b_var(t_idx_a, m_idx, start_time)
                            * b_var(
                                t_idx_b,
                                remaining_m_idx,
                                invalid_start_time
                            )
                        )

    # 1.4 Ensure optimal solutions are selected.
    D = 0
    machines = task.num_machines+1
    for t_idx in transaction_idxs:
        for m_idx in machine_idxs:
            for start_time in start_times(t_idx-1):
                w1 = machines**discrete_execution_length
                w2 = machines**(start_time + discrete_lengths[t_idx-1] - 1)
                w = w2/w1
                D += b_var(t_idx, m_idx, start_time)*w

    qubo = A + B + C + D
    model = qubo.compile()
    return model.to_qubo()


def convert_to_qiskit(
        input: Dict[Tuple[str, str], float],
        offset: float
        ) -> QuadraticProgram:
    """Convert QUBO to Qiskit.

    Converts the given QUBO representation in a format readable by Qiskit.

    Args:
        input: Dictionary containing
            key: Tuple of two binary variables as strings.
            value: Factor of the variable combination.
        offset: Constant offset of the problem.

    Returns:
        Representation of the problem as QUBO.
    """
    linear = {a: val for (a, b), val in input.items() if a == b}
    quadratic = {(a, b): val for (a, b), val in input.items() if a != b}

    qubo = QuadraticProgram()
    for var in linear.keys():
        qubo.binary_var(var)
    qubo.minimize(constant=offset, linear=linear, quadratic=quadratic)
    return qubo
