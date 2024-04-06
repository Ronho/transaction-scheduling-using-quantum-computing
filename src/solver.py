import numpy as np
import time

from dataclasses import dataclass
from dwave.samplers import SimulatedAnnealingSampler
from functools import wraps
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from src.qubo import convert_to_qiskit
from typing import Callable, Dict, List, Tuple


@dataclass
class Result:
    """Representation of a Result

    Attributes:
        active_vars: List of the activated variables.
        energy: Energy of the Hamiltonian for the given variables.
    """
    active_vars: List[str]
    energy: float


def _measure_time(func: Callable) -> Callable:
    """Decorator that takes the time a function requires.

    The result of the call to the returned function consists of a tuple
    containing the result of the original function as the first argument and
    the execution time as the second argument.

    Args:
        func (Callable): The function for which the time is to be taken.

    Returns:
        Callable: Wrapped function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        return res, end - start
    return wrapper


@_measure_time
def simulated_annealing_solver(
        qubo: Dict[Tuple[str, str], float],
        offset: float
        ) -> Result:
    """Solve a QUBO using Simulated Annealing.

    Args:
        qubo: Representation of the QUBO.
        offset: Offset of the QUBO.

    Returns:
        Result object containing the best variables that are activated and the
        energy including the offset.
    """
    simulated_annealing_sampler = SimulatedAnnealingSampler()
    sampleset = simulated_annealing_sampler.sample_qubo(qubo, num_reads=500)
    best_samples = sampleset.aggregate().lowest().data()
    best = max(best_samples, key=lambda x: x.num_occurrences)
    vars = [var for var, is_on in best.sample.items() if is_on == 1]
    return Result(
        active_vars=vars,
        energy=best.energy + offset
    )


@_measure_time
def sampling_vqe_solver(
        qubo: Dict[Tuple[str, str], float],
        offset: float
        ) -> Result:
    """Solve a QUBO using VQE.

    Args:
        qubo: Representation of the QUBO.
        offset: Offset of the QUBO.

    Returns:
        Result object containing the best variables that are activated and the
        energy including the offset.
    """
    ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
    svqe = SamplingVQE(sampler=Sampler(), ansatz=ansatz, optimizer=COBYLA())
    opt = MinimumEigenOptimizer(svqe)
    quadratic_program = convert_to_qiskit(qubo, offset)
    result = opt.solve(quadratic_program)
    return Result(
        active_vars=list(np.array(result.variable_names)[result.x == 1]),
        energy=result.fval
    )


@_measure_time
def qaoa_solver(qubo: Dict[Tuple[str, str], float], offset: float) -> Result:
    """Solve a QUBO using QAOA.

    Args:
        qubo: Representation of the QUBO.
        offset: Offset of the QUBO.

    Returns:
        Result object containing the best variables that are activated and the
        energy including the offset.
    """
    qaoa = QAOA(
        sampler=Sampler(),
        optimizer=COBYLA(),
        initial_point=[0.0, 0.0]
    )
    opt = MinimumEigenOptimizer(qaoa)
    quadratic_program = convert_to_qiskit(qubo, offset)
    result = opt.solve(quadratic_program)
    return Result(
        active_vars=list(np.array(result.variable_names)[result.x == 1]),
        energy=result.fval
    )


@_measure_time
def exact_solver(qubo: Dict[Tuple[str, str], float], offset: float) -> Result:
    """Solve a QUBO using an exact Solver.

    Args:
        qubo: Representation of the QUBO.
        offset: Offset of the QUBO.

    Returns:
        Result object containing the best variables that are activated and the
        energy including the offset.
    """
    exact = NumPyMinimumEigensolver()
    opt = MinimumEigenOptimizer(exact)
    quadratic_program = convert_to_qiskit(qubo, offset)
    result = opt.solve(quadratic_program)
    return Result(
        active_vars=list(np.array(result.variable_names)[result.x == 1]),
        energy=result.fval
    )
