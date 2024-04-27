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

    def csr_to_dense_coordinates(self, indptr, indices):
        #data, indptr, indices = compress_sparse_matrix(matrix)
        coordinates = []
        row_index = 0
        for i in range(len(indptr) - 1):
            for j in range(indptr[i], indptr[i + 1]):
                col_index = indices[j]
                coordinates.append((row_index, col_index))
            row_index += 1
        return coordinates
    
    def valid_convolution_pairs(self, input_coords, weight_coords, weight_dim, input_dim):
        valid_operations = []
        weight_rows, weight_cols  = weight_dim
        input_rows, input_cols = input_dim
        
        for input_coord in input_coords:
            input_row, input_col = input_coord
            for weight_coord in weight_coords:
                weight_row, weight_col = weight_coord
                
                # Calculate the start and end indices of the convolution window
                conv_window_start_row = input_row - weight_row
                conv_window_end_row = conv_window_start_row + weight_rows
                conv_window_start_col = input_col - weight_col
                conv_window_end_col = conv_window_start_col + weight_cols
                
                # Check if the convolution window fits within the input dimensions
                if (conv_window_start_row >= 0 and conv_window_end_row <= input_rows and
                    conv_window_start_col >= 0 and conv_window_end_col <= input_cols):
                    valid_operations.append((input_coord, weight_coord))
        
        return valid_operations
        
    def find_convolution_output_coordinates(self, valid_pairs):
        output_coords = []
        for input_coord, weight_coord in valid_pairs:
            input_row, input_col = input_coord
            weight_row, weight_col = weight_coord
            output_row = input_row - weight_row
            output_col = input_col - weight_col
            output_coords.append((output_row, output_col))
        return output_coords

