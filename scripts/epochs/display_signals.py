import pickle
import matplotlib.pyplot as plt

from bci3wads.features import epoch
from bci3wads.utils import constants

with open(constants.REF_EPOCH_PATH, 'rb') as f:
    data = pickle.load(f)

my_epoch = epoch.Epoch(data)

match_signals = my_epoch.get_match_signals()
mismatch_signals = my_epoch.get_mismatch_signals()

# Check if we can see any significant difference between a match and a mismatch
# signal.
plt.plot(match_signals[0, 3])
plt.plot(mismatch_signals[0, 0])
plt.show()
