import numpy as np

def read_off(path):
    # The function receives a path of a file, and given an .off file, it returns a tuple of two ndarrays,
    # one of the vertices, and one of the faces of the given mesh.

    with open(path, 'r') as file:
        f_lines = file.readlines()
        if f_lines[0] != 'OFF\n':
            print('Invalid OFF file - missing keyword "OFF"!')
            return
        [vertices_num, faces_num, _] = [int(n) for n in (f_lines[1].split())]
        v = [[float(part) for part in (line.split())] for line in f_lines[2:2+vertices_num]]
        f = [[int(part) for part in (line.split())] for line in f_lines[2+vertices_num:2+vertices_num+faces_num]]
        v = np.array(v, dtype=float)
        f = np.array(f, dtype=int)

    return v, f