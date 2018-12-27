from collections import Counter
from .qasm_parser_backend_generator import generate_backend

def qasm_runner_qiskit(qasm, ibmq_backend, shots=None, **kwargs):
    try:
        import qiskit
    except ImportError:
        raise ImportError("Cannot import qiskit. To use this backend, please install qiskit." +
                              " `pip install qiskit`")
    except Exception as e:
        if get_exception:
            return e
        raise ValueError("Unknown error raised when importing qiskit. " +
                "To get exception, run this backend with arg `get_exception=True`")
    else:
        if shots is None:
            shots = 1024
        qk_circuit = qiskit.load_qasm_string(qasm)
        result = qiskit.execute(qk_circuit, backend=ibmq_backend, shots=shots, **kwargs).result()
        counts = Counter({bits[::-1]: val for bits, val in result.get_counts().items()})
        return counts

ibmq_backend = generate_backend(qasm_runner_qiskit)
