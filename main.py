import numpy as np
from scipy.linalg import hankel, svd


class CustomNFourSID:
    def __init__(self, output_data, sys_order):
        """
        Initializes the CustomNFourSID class with output data and system order.

        :param output_data: The output data (y) from the system.
        :param sys_order: The order of the system to be identified.
        """
        self.y = np.asarray(output_data)  # Ensure the data is a numpy array
        self.sys_order = sys_order
        self.A = None
        self.C = None
        self.K = None

    def create_hankel_matrix(self, data, block_size):
        """
        Creates the Hankel matrix for the given data and block size.

        :param data: The input data to form the Hankel matrix.
        :param block_size: The size of the block (related to system order).
        :return: The Hankel matrix.
        """
        data = np.asarray(data)
        return hankel(data[:block_size], data[block_size - 1:])

    def subspace_identification(self):
        """
        Performs subspace system identification using the provided output data.
        Identifies the system matrices A, C, and Kalman gain K.

        :return: A dictionary containing matrices A, C, and K.
        """
        block_size = self.sys_order * 2  # Block size based on system order
        H = self.create_hankel_matrix(self.y, block_size)

        U, S, Vh = svd(H)

        U1 = U[:, :self.sys_order]
        S1 = np.diag(S[:self.sys_order])

        X = S1 @ U1.T
        self.A = X[:, 1:] @ np.linalg.pinv(X[:, :-1])
        self.C = U1[0, :]  # Output matrix for sequential data

        estimated_output = self.C @ X[:, :-1]
        estimated_output = estimated_output.flatten()[:len(self.y) - 1]

        pinv_X = np.linalg.pinv(X[:, :-1])
        self.K = (self.y[1:estimated_output.shape[0] + 1] - estimated_output) @ pinv_X[:len(
            self.y[1:estimated_output.shape[0] + 1]), :]

        return {
            'A': self.A,
            'C': self.C,
            'K': self.K
        }

    def concatenate_matrices_as_row(self):
        """
        Concatenates the matrices A, C, and K into a single row vector.

        :return: A single row vector containing all the elements of A, C, and K in order.
        """
        if self.A is None or self.C is None or self.K is None:
            raise ValueError("Subspace identification must be performed before concatenating matrices.")

        A_flat = self.A.flatten(order='C')  # Flatten A row-wise
        C_flat = self.C.flatten(order='C')  # Flatten C row-wise
        K_flat = self.K.flatten(order='C')  # Flatten K row-wise

        combined_row_vector = np.concatenate([A_flat, C_flat, K_flat])
        return combined_row_vector

    def predict_output(self, input_data):
        """
        Predicts the output for the given input data based on the identified system.

        :param input_data: The input data to the system.
        :return: The predicted output data.
        """
        if self.A is None or self.C is None or self.K is None:
            raise ValueError("Subspace identification must be performed before predicting output.")

        input_data = np.asarray(input_data)
        X = np.zeros((self.sys_order, len(input_data)))

        for t in range(1, len(input_data)):
            X[:, t] = self.A @ X[:, t - 1] + self.K @ input_data[t - 1]

        predicted_output = self.C @ X
        return predicted_output.flatten()

    def set_matrices(self, A, C, K):
        """
        Allows setting the matrices A, C, and K manually.

        :param A: State transition matrix.
        :param C: Output matrix.
        :param K: Kalman gain matrix.
        """
        self.A = np.asarray(A)
        self.C = np.asarray(C)
        self.K = np.asarray(K)

    def get_matrices(self):
        """
        Returns the identified system matrices A, C, and K.

        :return: A dictionary containing matrices A, C, and K.
        """
        return {
            'A': self.A,
            'C': self.C,
            'K': self.K
        }
