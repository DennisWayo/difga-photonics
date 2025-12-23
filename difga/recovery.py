import pennylane as qml

def recovery_layer_mm(params):
    phi0, d0r, d0i, phi1, d1r, d1i = params
    qml.Rotation(phi0, wires=0)
    qml.Displacement(d0r + 1j * d0i, 0.0, wires=0)
    qml.Rotation(phi1, wires=1)
    qml.Displacement(d1r + 1j * d1i, 0.0, wires=1)

def recovery_layer_sm(params):
    phi, dr, di = params
    qml.Rotation(phi, wires=0)
    qml.Displacement(dr + 1j * di, 0.0, wires=0)