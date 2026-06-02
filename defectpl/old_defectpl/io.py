import json
import numpy as np


def read_properties(filename):
    """
    Read the properties from the json file.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    intensity = np.array(
        [complex(real, imag) for real, imag in data["I"]], dtype=np.complex128
    )
    properties = {
        "frequencies": np.array(data["frequencies"]),
        "iprs": np.array(data["iprs"]),
        "localization_ratio": np.array(data["localization_ratio"]),
        "qks": np.array(data["qks"]),
        "Sks": np.array(data["Sks"]),
        "S_omega": np.array(data["S_omega"]),
        "omega_range": np.array(data["omega_range"]),
        "I": intensity,
        "resolution": np.array(data["resolution"]),
        "delta_R": np.array(data["delta_R"]),
        "delta_Q": np.array(data["delta_Q"]),
        "HR_factor": np.array(data["HR_factor"]),
        "dR": np.array(data["dR"]),
        "EZPL": np.array(data["EZPL"]),
        "gamma": np.array(data["gamma"]),
        "natoms": np.array(data["natoms"]),
        "masses": np.array(data["masses"]),
        "max_energy": np.array(data["max_energy"]),
    }
    return properties
