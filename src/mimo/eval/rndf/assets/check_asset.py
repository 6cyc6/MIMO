import numpy as np

data = np.load("bad_bowls_old.npz", allow_pickle=True)

bad_list = data["bad_ids"].tolist()
bad_list.append("e4c871d1d5e3c49844b2fa2cac0778f5")
np.savez("bad_bowls", bad_ids=bad_list)

# data = np.load("bad_bowls.npz", allow_pickle=True)
# bad_list = data["bad_ids"]
# print(bad_list)
