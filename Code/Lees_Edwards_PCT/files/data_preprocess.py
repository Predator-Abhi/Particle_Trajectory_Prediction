import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import root_mean_squared_error
import tarfile
import h5py
import io

class Data_Preprocess:
    def __init__(self, path, file_n=0, init_chop=20):
        super(Data_Preprocess, self).__init__()
        self.PATH = path
        self.FILE_N = file_n
        self.INIT_CHOP = init_chop
        return 
        
    def get_pos(self, h5file):
        frames = len(h5file)
        lists_t = []
        lists_id = []
        lists_x = []
        lists_y = []
        lists_z = []
        for frame in range(frames):
            step = h5file['Step#'+str(frame)]
            
            # get time of frame
            simulation_time = step.attrs['Time'][0]
            lists_t.append(simulation_time)
        
            # get particle ids
            particle_id = step['Identifier'][:]
            lists_id.append(np.expand_dims(np.array(particle_id), 1))
            
            # get corners
            lower_corner = step.attrs['Lower']
            upper_corner = step.attrs['Upper']
            
            # get particle positions
            particle_pos_x = step['Coords_0'][:]
            particle_pos_y = step['Coords_1'][:]
            particle_pos_z = step['Coords_2'][:]
            lists_x.append(np.expand_dims(np.array(particle_pos_x), 1))
            lists_y.append(np.expand_dims(np.array(particle_pos_y), 1))
            lists_z.append(np.expand_dims(np.array(particle_pos_z), 1))  
    
        time = np.array(lists_t)
        a = np.array(lists_id)
        b = np.array(lists_x)
        c = np.array(lists_y)
        d = np.array(lists_z)
        
        out = np.concatenate((a, b, c, d), axis=2)
        
        return time, upper_corner, lower_corner, out
    
    def extraction(self):
        tar = tarfile.open(self.PATH, "r:gz")
        lists_data = []
        for t in tar.getmembers():
            temp = tar.extractfile(t)
            temp = io.BytesIO(temp.read())
            temp = h5py.File(temp, "r")
            time, upper_corner, lower_corner, data = self.get_pos(temp)
            lists_data.append([time, upper_corner, lower_corner, data])
            break
        
        time = lists_data[self.FILE_N][0][self.INIT_CHOP:]
        upper_corner = lists_data[self.FILE_N][1]
        lower_corner = lists_data[self.FILE_N][2]
        data = lists_data[self.FILE_N][3][self.INIT_CHOP:]
        
        
        box_length = 2*abs(lower_corner)
        
        # sorting the data according to ids
        temp = np.zeros_like(data)
        for timestep in range(data.shape[0]):
            for particle in range(data.shape[1]):
                original_id = int(data[timestep, particle, 0])  # Get the original ID
                temp[timestep, original_id, :] = data[timestep, particle, :]  # Reorder the particle
        data = temp[:, :, 1:]
        
        return time, box_length, data
    
    def get_data(self, output_timesteps):
        time, box_length, data = self.extraction()
        t = data.shape[0]
        X = np.expand_dims(data[:(t-output_timesteps)], 0)
        y = np.expand_dims(data[-output_timesteps:], 0)
        X = np.transpose(X, (0, 3, 1, 2)) # X shape (n, c, t, p)
        y = np.transpose(y, (0, 2, 1, 3)) # y shape (n, p, t, c)
        time_X = time[:(t-output_timesteps)]
        time_y = time[-output_timesteps:]
        return X, y, time_X, time_y, box_length
    
    def get_lee_edward_distance(self, r_a, r_b, box_length, shear_rate, time):
    
        r = r_a - r_b
        # rel_velocity = v_a - v_b
        
        # Calculate the amount of periodic wrapping required
        lee_edwards_velocity = shear_rate * box_length[0]
        lee_edwards_shift = lee_edwards_velocity * time
        lee_edwards_shift -= np.round(lee_edwards_shift/box_length[2]) * box_length[2]
        
        # Apply periodic boundary conditions to position and velocity
        if r[0] < -0.5*box_length[0]:
            r[0] += box_length[0]
            r[2] -= lee_edwards_shift
            # rel_velocity[2] -= lee_edwards_velocity  
        elif r[0] >= 0.5*box_length[0]:
            r[0] -= box_length[0]
            r[2] += lee_edwards_shift
            # rel_velocity[2] += lee_edwards_velocity
            
        if r[1] < -0.5*box_length[1]:
            r[1] += box_length[1]
        elif r[1] >= 0.5*box_length[1]:
            r[1] -= box_length[1]
            
        if r[2] < -0.5*box_length[2]:
            r[2] += box_length[2]
        elif r[2] >= 0.5*box_length[2]:
            r[2] -= box_length[2]
        
        return r #rel_velocity
    
    def get_G_fixed(self, X, box_length, time_X, d_threshold, shear_rate=1000, alpha = 0.001):
        # Compute matrix
        # X is a shape of (n, c, t, p)
        time = time_X[-1]
        temp = X[0, :, -1, :] #temp is of shape (c, p)
        particles = np.transpose(temp, (1, 0)) # matrix should be the shape of (p, c)
        n_particles = particles.shape[0]
        
        distance_matrix = np.zeros((n_particles, n_particles))
        for i in range(n_particles):
            for j in range(n_particles):
                r = self.get_lee_edward_distance(particles[i], particles[j], box_length, shear_rate, time)
                distance_matrix[i, j] = np.sqrt(np.sum(r**2))
        
        # compute A0, A1 matrices
        A1 = (distance_matrix < d_threshold).astype(int)
        A0 = np.eye(n_particles, dtype='int')
        
        # Compute lambdas
        Lambda_0 = np.sum(A0, axis=1) + alpha
        Lambda_1 = np.sum(A1, axis=1) + alpha
        
        # Compute G_fixed
        t = np.diag(1/np.sqrt(Lambda_0))
        G0_fixed = np.matmul(t, np.matmul(A0, t))
        
        t = np.diag(1/np.sqrt(Lambda_1))
        G1_fixed = np.matmul(t, np.matmul(A1, t))
        
        G_fixed = G0_fixed + G1_fixed
        G_fixed = np.expand_dims(G_fixed, 0) # G_fixed should be of shape (n, p, p)
        
        return G_fixed
    
    def transform_data(self, X, y, box_length=None):
        a = box_length[0]
        # X shape (n, c, t, p)
        X_s = np.sin(2*np.pi*X/a)
        X_c = np.cos(2*np.pi*X/a)
        X_sc = np.concatenate((X_s[:, 0:1], X_c[:, 0:1],
                               X_s[:, 1:2], X_c[:, 1:2],
                               X_s[:, 2:3], X_c[:, 2:3]), 1)
        y_s = np.sin(2*np.pi*y/a)
        y_c = np.cos(2*np.pi*y/a)
        y_out = np.concatenate((y_s[:, :, :, 0:1], y_c[:, :, :, 0:1],
                               y_s[:, :, :, 1:2], y_c[:, :, :, 1:2],
                               y_s[:, :, :, 2:3], y_c[:, :, :, 2:3]), -1)
        dX = X_sc[:, :, 1:] - X_sc[:, :, :-1]
        last_location = X_sc[:, :, -1, :] # has a shape of (n, c, p)
        last_location = np.transpose(last_location, (0, 2, 1)) # make shape (n, p, c)
        return dX, last_location, y_out
            
    def inverse_transform(self, y_pred, box_length=None):
        a = box_length[0]
        y_x = torch.arctan2(y_pred[:, :, :, 0:1], y_pred[:, :, :, 1:2])
        y_y = torch.arctan2(y_pred[:, :, :, 2:3], y_pred[:, :, :, 3:4])
        y_z = torch.arctan2(y_pred[:, :, :, 4:5], y_pred[:, :, :, 5:6])
        y_c = torch.concatenate((y_x, y_y, y_z), -1)
        out = a*y_c/(2*torch.pi)
        return out

    def compute_rmse(self, y_true, y_pred, box_length, time_y, shear_rate=1000):
        # y shape (n, p, t, c)
        rmse = 0
        (a, b, c, d) = y_true.shape
        for i in range(a):
            for j in range(c):
                for k in range(b):
                    r = self.get_lee_edward_distance(y_true[i, k, j, :], y_pred[i, k, j, :], box_length, shear_rate, time_y[j])
                    rmse += np.sum(r**2)
        N = a*b*c
        rmse = np.sqrt(rmse/N)
        print(f"RMSE: {rmse}")
        return rmse