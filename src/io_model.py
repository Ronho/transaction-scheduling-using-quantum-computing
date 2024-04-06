from pydantic import BaseModel, field_validator
from typing import List, Type


class DataModel(BaseModel):
    """Representation of the data for easy reading and writing to disk.

    Attributes:
        dataset: Name of the dataset.
        task_id: Identifier of the task.
        method: Method used to get this result.
        energy: Energy of the Hamiltonian for the given variables rounded to
            the fourth decimal position.
        exec_time: Execution time rounded to the fourth decimal position.
        vars: List of the activated variables.
    """

    dataset: str
    task_id: int
    method: str
    energy: float
    exec_time: float
    vars: List[str]

    @field_validator('energy', 'exec_time')
    @classmethod
    def round_float(cls, val: float) -> float:
        """Validator to round floats to fourth decimal position.

        Args:
            val: Value to round.

        Returns:
            Rounded value.
        """
        return round(val, 4)


def append_data_to_file(file_path: str, string: str) -> None:
    """Append a string to a file.

    The function appends a string. If the file does not exist.

    Args:
        file_path: Path to the file.
        data: Dictionary to be appended.
    """
    with open(file_path, 'a', newline='') as file:
        file.write(string + '\n')


def read_json_lines_file_to_pydantic(
        file_path: str,
        model: Type[BaseModel]
        ) -> List[BaseModel]:
    """Read a JSON lines file and convert each line to a Pydantic object.

    Args:
        file_path: Path to the JSON lines file.
        model: Pydantic class to convert each line into.

    Returns:
        List containing each line as a Pydantic model.
    """
    entries = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            entry = model.model_validate_json(line)
            entries.append(entry)
    return entries
