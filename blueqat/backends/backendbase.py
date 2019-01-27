"""
`gate` module implements quantum gate operations.
This module is internally used.
"""

from abc import ABC
import copy

class Backend(ABC):
    """Abstract quantum gate processor backend class."""
    def copy(self):
        """Returns (deep)copy of Backend.

        Backend developer must support `copy` method.
        If required, the developer can override this method.
        """
        return copy.deepcopy(self)

    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        """Preprocess of backend run.
        Backend developer can override this function.
        """
        return gates, None

    def _postprocess_run(self, ctx):
        """Postprocess of backend run
        Backend developer can override this function.
        """
        return None

    def _run_gates(self, gates, n_qubits, ctx):
        """Iterate gates and call backend's action for each gates"""
        for gate in gates:
            action = self._get_action(gate)
            if action is not None:
                ctx = action(gate, ctx)
            else:
                ctx = self._run_gates(gate.fallback(n_qubits), n_qubits, ctx)
        return ctx

    def _run(self, gates, n_qubits, args, kwargs):
        """Default implementation of `Backend.run`.
        Backend developer shouldn't override this function, but override `run` instead of this.

        The default flow of running is:
            1. preprocessing
            2. call the gate action which defined in backend
            3. postprocessing

        Backend developer can:
            1. Define preprocessing process. Override `_preprocess_run`
            2. Define the gate action. Define methods `gate_{gate.lowername}`,
               for example, `gate_x` for X gate, `gate_cx` for CX gate.
            3. Define postprocessing process (and make return value). Override `_postprocess_run`
        Otherwise, the developer can override `run` method if they want to change the flow of run.
        """
        gates, ctx = self._preprocess_run(gates, n_qubits, args, kwargs)
        self._run_gates(gates, n_qubits, ctx)
        return self._postprocess_run(ctx)

    def make_cache(self, gates, n_qubits):
        """Make internal cache to reduce the time of running the circuit.

        Some backends may implement this method. Otherwise, this method do nothing.
        This is a temporary API and may changed or deprecated in future."""
        return None

    def run(self, gates, n_qubits, *args, **kwargs):
        """Run the backend."""
        return self._run(gates, n_qubits, args, kwargs)

    def _get_action(self, gate):
        try:
            return getattr(self, "gate_" + gate.lowername)
        except AttributeError:
            return None

    def _has_action(self, gate):
        return hasattr(self, "gate_" + gate.lowername)

    def _resolve_fallback(self, gates, n_qubits):
        """Resolve fallbacks and flatten gates."""
        flattened = []
        for g in gates:
            if self._has_action(g):
                flattened.append(g)
            else:
                flattened += self._resolve_fallback(g.fallback(n_qubits), n_qubits)
        return flattened
