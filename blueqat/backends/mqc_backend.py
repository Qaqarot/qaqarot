import json
import math
import random
import urllib.request
from collections import Counter, namedtuple

import numpy as np

from ..gate import *
from .backendbase import Backend

_MQCContext = namedtuple("MQCContext", "ops n_qubits token shots returns url")
DEFAULT_URL = "https://api.mdrft.com/v1/command"

class MQCError(Exception):
    def __init__(self, err):
        self.err_type = err.__class__.__name__
        self.err = err
        self.statuscode = None
        self.err_json = None
        if isinstance(err, urllib.request.HTTPError):
            self.statuscode = err.code
            self.err_msg = err.read().decode(encoding="utf-8", errors="backslashreplace")
            try:
                self.err_json = json.loads(self.err_msg)
            except json.JSONDecodeError:
                pass
        else:
            self.err_msg = str(err)

    def __str__(self):
        s = self.err_type + ", "
        if self.statuscode is not None:
            s += "Status Code: " + str(self.statuscode) + ", "
        s += self.err_msg
        return s

class MQCBackend(Backend):
    """Backend for MDR Quantum Cloud."""
    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        def _parse_args(token, shots=1024, returns="shots", url=DEFAULT_URL, **_kwargs):
            if returns not in ("shots", "_res", "_urllib_req", "_urllib_res", "_urllib_req_res"):
                raise ValueError(f"Unknown returns type '{returns}'")
            return token, shots, returns, url
        token, shots, returns, url = _parse_args(*args, **kwargs)
        return gates, _MQCContext([], n_qubits, token, shots, returns, url)

    def _postprocess_run(self, ctx):
        req = urllib.request.Request(
            ctx.url,
            json.dumps({
                "access_token": ctx.token,
                "command": "createTask",
                "payload": {
                    "shots": ctx.shots,
                    "ops": ctx.ops,
                }
            }).encode('utf-8'),
            method="POST",
            headers={
                "Content-Type": "application/json",
            }
        )
        if ctx.returns == "_urllib_req":
            return req
        try:
            with urllib.request.urlopen(req) as res:
                if ctx.returns == "_urllib_res":
                    return res
                if ctx.returns == "_urllib_req_res":
                    return req, res
                result = json.loads(res.read())
                if ctx.returns == "_res":
                    return result
                assert ctx.returns == "shots"
                return Counter(result["mqc_result"])
        except urllib.request.HTTPError as e:
            if ctx.returns == "_urllib_res":
                return e.fp
            if ctx.returns == "_urllib_req_res":
                return req, e.fp
            raise MQCError(e)
        except urllib.request.URLError as e:
            raise MQCError(e)


    def _one_qubit_gate_noargs(self, gate, ctx):
        for idx in gate.target_iter(ctx.n_qubits):
            ctx[0].append({gate.uppername: [idx]})
        return ctx

    gate_x = _one_qubit_gate_noargs
    gate_y = _one_qubit_gate_noargs
    gate_z = _one_qubit_gate_noargs
    gate_h = _one_qubit_gate_noargs
    gate_t = _one_qubit_gate_noargs
    gate_s = _one_qubit_gate_noargs

    def _two_qubit_gate_noargs(self, gate, ctx):
        for control, target in gate.control_target_iter(ctx[1]):
            ctx[0].append({gate.uppername: [control, target]})
        return ctx

    gate_cz = _two_qubit_gate_noargs
    gate_cx = _two_qubit_gate_noargs

    def _one_qubit_gate_args_theta(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append({gate.uppername: [idx, gate.theta]})
        return ctx

    gate_rx = _one_qubit_gate_args_theta
    gate_ry = _one_qubit_gate_args_theta
    gate_rz = _one_qubit_gate_args_theta

    def gate_u1(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append({gate.uppername: [idx, gate.lambd]})

    def gate_u2(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append({gate.uppername: [idx, gate.phi, gate.lambd]})

    def gate_u3(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append({gate.uppername: [idx, gate.theta, gate.phi, gate.lambd]})

    def gate_measure(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append({"M": [idx]})
        return ctx
