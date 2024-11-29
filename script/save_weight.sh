import numpy as np
np.save('b_splines.npy', self.b_splines(x).view(x.size(0), -1).cpu().numpy(), allow_pickle=True)