import numpy as np

def basic_meshgrid(x, y):
        """
        This method is only made for my own understanding of what a mesh is, and why we need meshes
        when creating 3D-plots.
        """

        ### This is a verbose version of what the mesh does:

        """
        X_grid = []
        Y_grid = []

        for _ in range(len(y)):
            X_grid.append([x[j] for j in range(len(x))])
        
        for _ in range(len(x)):
            Y_grid.append([y[j] for j in range(len(y))])
        
        Y_grid = list(map(list, zip(*Y_grid)))
        """

        # What we essentially want is a len(y) * len(x) matrix, and this can be achieved
        # by doing for-looping, first looping through len(y) and then len(x) in inner loop.

        # For the x_grid, this works just fine. 
        # It gets a little more complicated for the y_grid.
        # Instead of making a len(y) by len(x) matrix directly, what we do is that we make
        # a len(x) by len(y) grid, and then we transpose it.

        # The same method, written with list comprehension. 

        # Alternative method of creating Y with the use of zip to transpose the matrix.
        # Y_grid = list(map(list, zip(*[[y[j] for j in range(len(y))] for _ in range(len(x))])))
        
        X_grid = np.array([[x_val for x_val in x] for _ in range(len(y))])
        Y_grid = np.array([[y_val for _ in range(len(x))] for y_val in y])
        return X_grid, Y_grid