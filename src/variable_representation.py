from dataclasses import dataclass

@dataclass
class VariableRepresentation:
    """Container for easier access of the variables.

    Attributes:
        transaction_id: Id of the transaction.
        machine_id: Id of the machine.
        start_time: Start time of the transaction.
    """
    transaction_id: int
    machine_id: int
    start_time: int

    @classmethod
    def from_string(cls, input: str) -> "VariableRepresentation":
        """Initialize a VariableRepresentation from a string.

        The string must follow the concept
        "transaction_id-machine_id-start_time". transaction_id and machine_id
        are expected to start from 1. This is corrected backwards to 0 within
        this method.

        Args:
            input: String to use.

        Returns:
            VariableRepresentation object.
        """
        splits = input.split('-')
        return cls(
            transaction_id=int(splits[0])-1,
            machine_id=int(splits[1])-1,
            start_time=int(splits[2])
        )
