JSON schema specifications of blueqat Circuit
=============================================

This documentations describe the JSON format of ``blueqat.Circuit``.

Motivations
-----------

When you want to send or receive the Circuit to/from remote servers, `serialization<https://en.wikipedia.org/wiki/Serialization>` is an effective way.
`Pickling<https://docs.python.org/3/library/pickle.html>` is one of the traditional serializations in Python language. However, pickling is insecure when you communicate with unreliable servers or clients.  Recently, JSON has been widely used as serializing format with unspecified machines.
Therefore, we make specifications of blueqat Circuit JSON formats.

Serialize and deserialize function
----------------------------------

``blueqat.circuit_func.json_serializer.serialize`` and ``blueqat.circuit_func.json_serializer.deserialize`` are provided.


Version 1
---------

JSON schema version 1 is available from blueqat version 0.4.6.

.. code-block
  Circuit := { "schema": Schema, "n_qubits": int, "ops": [Op...] }
  Schema := { "name": "blueqat-circuit", "version": "1" }
  Op := { "name": lowername-of-operation, "targets": [int...], "params": [number...] }
  lower-name-of-operation := the name of gates, operations. e.g. "x", "h", "cx", "measure", ...
  int:= integer number
  number:= number
