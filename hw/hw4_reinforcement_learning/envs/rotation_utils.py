import numpy as np
import mujoco


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q_out = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(q_out, q1, q2)
    return q_out


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q_conj = np.zeros(4, dtype=np.float64)
    mujoco.mju_negQuat(q_conj, q)
    return q_conj


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q_normalized = np.asarray(q, dtype=np.float64).copy()
    mujoco.mju_normalize4(q_normalized)
    return q_normalized


def rot_mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(-1))
    return quat