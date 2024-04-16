def csr(arr):
    value = []
    col_id = []
    row_ptr = []
    for i in range(len(arr)):
        new_row = True
        for j in range(len(arr[0])):
            if arr[i][j] != 0:
                if new_row:
                    new_row = False
                    value.append(arr[i][j])
                    col_id.append(j)
                    row_ptr.append(len(value) - 1)
                else:
                    value.append(arr[i][j])
                    col_id.append(j)
    return value, col_id, row_ptr

def get_coords(value, col_id, row_ptr):
    # loop the row_ptr
    for i in range(len(row_ptr)):
        # i = row of the element, col_id[j] = column of the element
        start = row_ptr[i]
        end = 0
        if i == len(row_ptr) - 1:
            end = len(value)
        else:
            end = row_ptr[i + 1]
        for j in range(start, end):
            print(f"Value: {value[j]}, Coordinates: ({i}, {col_id[j]})")

def multiply_csr(value1, col_id1, row_ptr1, value2, col_id2, row_ptr2):
    value, col_id, row_ptr = [], [], []
    for i1 in range(len(row_ptr1)):
        # i = row of the element, col_id[j] = column of the element
        
        start1 = row_ptr1[i1]
        end1 = 0
        if i1 == len(row_ptr1) - 1:
            end1 = len(value1)
        else:
            end1 = row_ptr1[i1 + 1]
        for j1 in range(start1, end1):
            # print(f"\nvalue1: {value1[j1]}, coord: ({i1}, {col_id1[j1]})")
            for i2 in range(len(row_ptr2)):
                start2 = row_ptr2[i2]
                end2 = 0
                if i2 == len(row_ptr2) - 1:
                    end2 = len(value2)
                else:
                    end2 = row_ptr2[i2 + 1]
                for j2 in range(start2, end2):
                    # print(f"value2: {value2[j2]}, cord: ({i2}, {j2})")
                    if col_id1[j1] == i2:
                        value.append(value1[j1] * value2[j2])
                        col_id.append(col_id2[j2])
    print(value)
    print(col_id)

def decode_csr(value, col_id, row_ptr):

    # TODO: Add row_ptr. The order of elements in the output csr is not maintained.
    vals = dict()
    for i in range(len(row_ptr)):
        start = row_ptr[i]
        end = 0
        if i == len(row_ptr) - 1:
            end = len(value)
        else:
            end = row_ptr[i + 1]
        for j in range(start, end):
            vals[(i, col_id[j])] = value[j]
    arr = list()
    for i in range(len(row_ptr)):
        l = []
        for j in range(len(row_ptr)):
            if vals.get((i, j), 0) == 0:
                l.append(0)
            else:
                l.append(vals[(i, j)])
        arr.append(l)
    return arr

def print_arr(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            print(f"{arr[i][j]} ", end='')
        print("")

if __name__ == "__main__":
    ip = [
        [0,1,0],
        [1,0,2],
        [3,0,0]
    ]

    filter = [
        [0,0,1],
        [1,0,1],
        [2,1,0]
    ]

    v1, c1, r1 = csr(ip)
    print("Input\n")
    print(f"value: {v1}")
    print(f"col_id: {c1}")
    print(f"row_ptr: {r1}")

    v2, c2, r2 = csr(filter)
    print("Filter\n")
    print(f"value: {v2}")
    print(f"col_id: {c2}")
    print(f"row_ptr: {r2}")

    # get_coords(v, c, r)
    multiply_csr(v1, c1, r1, v2, c2, r2)
    
    decoded_arr = decode_csr(v1, c1, r1)
    # print(decoded_arr)
    print_arr(decoded_arr)