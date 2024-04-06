import numpy as np

from dataclasses import dataclass
from pathlib import Path
from src.task import TransactionScheduleTask


@dataclass
class Dataset:
    """Container for Transaction Scheduling Tasks

    Attributes:
        name: Name of the dataset.
        tasks: Transaction scheduling tasks.
    """
    name: str
    tasks: np.ndarray


def read_dataset(path: Path) -> Dataset:
    """Read a dataset from a file.

    The filename must be of the format
    "SchedulingWithConflicts_n<NUMBER_OF_TRANSACTIONS>_m<NUMBER_OF_CORES>", for
    example, "SchedulingWithConflicts_n8_m3".

    Each new line of the file must contain a single task. The following
    described elements are separared by an empty space. Spaces must not be used
    for anything else within the file. The first element represents the lengths
    comma separated. The second element represents the conflicts in a two
    dimensional array, also comma separated. The last element represents an
    optimal solution, which is given by a one dimensional array which is comma
    separated as well.

    Args:
        path: Location of the file to read the dataset from.

    Returns:
        A dataset containing the tasks.
    """
    if not path.is_file():
        raise ValueError('File does not exist: ', path)

    cores = int(path.stem.split('_m')[-1])
    tasks = []

    with path.open('r') as file:
        for id, line in enumerate(file.readlines()):
            lengths, conflicts, solution = line.split(' ')

            lengths = lengths.replace('[', '').replace(']', '')
            lengths = np.fromstring(lengths, dtype=np.float64, sep=',')

            tmp = []
            for arr in conflicts.split('],['):
                arr = arr.replace('[', '').replace(']', '')
                arr = np.fromstring(arr, dtype=np.uint8, sep=',')
                tmp.append(arr)
            conflicts = np.array(tmp)

            solution = solution.replace('[', '').replace(']', '')
            solution = np.fromstring(solution, dtype=np.uint8, sep=',')

            tasks.append(TransactionScheduleTask(
                id=id,
                num_machines=cores,
                lengths=lengths,
                conflicts=conflicts,
                solution=solution))

    return Dataset(name=path.stem, tasks=np.array(tasks))
