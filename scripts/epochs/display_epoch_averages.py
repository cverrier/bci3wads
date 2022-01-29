import pickle
import matplotlib.pyplot as plt

from bci3wads.features import epoch
from bci3wads.utils import constants

with open(constants.REF_EPOCH_PATH, 'rb') as f:
    data = pickle.load(f)

my_epoch = epoch.Epoch(data)

match_average = my_epoch.average_signals(cls='match')
mismatch_average = my_epoch.average_signals(cls='mismatch')

plt.plot(match_average, label='match')
plt.plot(mismatch_average, label='mismatch')
plt.legend()
plt.show()
