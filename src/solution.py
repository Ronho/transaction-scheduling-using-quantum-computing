import matplotlib.pyplot as plt

from dataclasses import dataclass
from matplotlib.patches import Rectangle
from typing import List, Iterable


@dataclass
class TransactionSolution:
    """Solution for one Transaction.

    Attributes:
        id: Identifier/index of the transaction.
        start_time: Time when the transaction starts running on the machine.
        end_time: Time when the transaction completes running on the machine.
        duration: Duration of the transaction.
    """
    id: int
    start_time: float
    end_time: float
    duration: float


@dataclass
class MachineSolution:
    """Solution for one Machine.

    Attributes:
        transactions: Transactions running on this machine.
    """
    transactions: List[TransactionSolution]

    @property
    def processing_time(self) -> float:
        """Processing time.

        The processing time is defined as the first time point that this
        machine can work on a new transaction.

        Returns:
            The current processing time of the machine.
        """
        if len(self.transactions) == 0:
            return 0
        return self.transactions[-1].end_time


@dataclass
class Solution:
    """Single Representation of a Solution.

    Contains detailed information about when a transaction starts and ends, and
    on which machine the transaction is running.

    Solutions are considered as non-mutable objects as changes to them could
    result in unrealistic representations.

    Attributes:
        machines: Solution objects per machine.
    """
    machines: List[MachineSolution]

    @property
    def length(self) -> float:
        """Length of the solution.

        The length of the solution is given by the maximum completion time of
        all transactions assuming that the first start time is 0.

        Returns:
            The length of the solution.
        """
        if hasattr(self, '_length'):
            return self._length

        lengths = [t.end_time for m in self.machines for t in m.transactions]
        self._length = max(lengths)
        return self._length

    def visualize(self) -> None:
        """Visualize the solution as a machine time diagram."""
        _, ax = plt.subplots()

        for id, machine in enumerate(self.machines):
            id += 1
            for transaction in machine.transactions:
                ax.add_patch(Rectangle(
                    (transaction.start_time, id-0.25),
                    transaction.duration, 0.5,
                    fill=False,
                    lw=2,
                    linestyle="dotted"
                ))

        plt.xlim(left=0, right=self.length)
        plt.ylim(bottom=0, top=id+1)
        plt.ylabel('Machines')
        plt.xlabel('Time')
        plt.show()

    def solution_to_solution_order(self) -> Iterable[int]:
        """Generate a solution order in processing time format.

        Returns:
            Solution order.
        """
        num_transactions = sum(len(m.transactions) for m in self.machines)
        # These machines are used to reconstruct the way the original machines
        # were built.
        machines = [MachineSolution([]) for _ in range(len(self.machines))]
        # Tracks the transaction index for the original machines to avoid
        # duplicating transactions.
        index_tracker = {m: 0 for m in range(len(self.machines))}

        solution_order = []
        for _ in range(num_transactions):
            machine = min(machines, key=lambda x: x.processing_time)
            machine_id = machines.index(machine)

            transaction_id = index_tracker[machine_id]
            transaction = self.machines[machine_id] \
                .transactions[transaction_id]

            solution_order.append(transaction.id)

            machine.transactions.append(transaction)
            index_tracker[machine_id] += 1
        return solution_order
