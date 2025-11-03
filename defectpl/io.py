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
        "frequencies": data["frequencies"],
        "iprs": data["iprs"],
        "localization_ratio": data["localization_ratio"],
        "qks": data["qks"],
        "Sks": data["Sks"],
        "S_omega": data["S_omega"],
        "omega_range": data["omega_range"],
        "I": intensity,
        "resolution": data["resolution"],
        "delta_R": np.array(data["delta_R"]),
        "delta_Q": data["delta_Q"],
        "HR_factor": data["HR_factor"],
        "dR": np.array(data["dR"]),
        "EZPL": data["EZPL"],
        "gamma": data["gamma"],
        "natoms": data["natoms"],
        "masses": data["masses"],
        "max_energy": data["max_energy"],
    }
    return properties
