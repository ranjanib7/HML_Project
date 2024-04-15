import numpy as np

class coordinate_compute_blk:
    def __init__(self) -> None:
        pass

    def find_output_coordinates(self, ifmap_op_mat, filter_op_mat):
        output_coordinates = []
        input_rows = len(ifmap_op_mat.input_indptr)-1
        input_cols = max(ifmap_op_mat.input_indices) + 1
        filter_size = len(filter_op_mat.filter_indptr) - 1
        stride = 1

        # Calculate the dimensions of the output matrix
        output_rows = (input_rows - filter_size + 1) // stride
        output_cols = (input_cols - filter_size + 1) // stride

        # Iterate over each position in the output matrix
        for i in range(output_rows):
            for j in range(output_cols):
                # Calculate the starting position in the input matrix for this output position
                start_row = i * stride
                start_col = j * stride

                # Add the output indices to the list
                output_indices = []
                for k in range(filter_size):
                    row_idx = start_row + k
                    row_start = ifmap_op_mat.input_indptr[row_idx]
                    row_end = ifmap_op_mat.input_indptr[row_idx + 1]
                    for l in range(filter_size):
                        col_idx = start_col + l
                        if col_idx in ifmap_op_mat.input_indices[row_start:row_end]:
                            output_indices.append((row_idx, col_idx))

                output_coordinates.append(output_indices)

        return output_coordinates