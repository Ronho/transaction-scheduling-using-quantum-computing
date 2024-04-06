from collections import defaultdict
from src.solution import Solution, MachineSolution
from src.task import TransactionScheduleTask
from src.utils import build_transaction_solution
from src.variable_representation import VariableRepresentation
from typing import Iterable


def binary_variables_to_solution(
        vars: Iterable[str],
        task: TransactionScheduleTask
        ) -> Solution:
    representations = [VariableRepresentation.from_string(var) for var in vars]
    machines = [MachineSolution([]) for _ in range(task.num_machines)]

    machine_var_map = defaultdict(list)
    [machine_var_map[rep.machine_id].append(rep) for rep in representations]
    {k: sorted(v, key=lambda x: x.start_time)
     for k, v in machine_var_map.items()}

    for _ in range(task.num_transactions):
        machine = min(machines, key=lambda x: x.processing_time)
        machine_id = machines.index(machine)

        transaction_var = machine_var_map[machine_id].pop()
        transaction_id = transaction_var.transaction_id
        duration = task.lengths[transaction_id]

        build_transaction_solution(
            machine=machine,
            machines=machines,
            transaction_id=transaction_id,
            duration=duration,
            conflicts=task.conflicts
        )

    return Solution(machines=machines)


def binary_variables_to_solution_order(
        vars: Iterable[str],
        task: TransactionScheduleTask
        ) -> Iterable[int]:
    solution = binary_variables_to_solution(vars=vars, task=task)
    return solution.solution_to_solution_order()
