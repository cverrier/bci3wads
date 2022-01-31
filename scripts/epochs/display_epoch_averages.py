import matplotlib.pyplot as plt

from bci3wads.utils import data
from bci3wads.features import epoch
from bci3wads.utils import constants

my_epoch = epoch.Epoch(data.load_pickle(constants.REF_EPOCH_PATH))

match_average = my_epoch.average_signals(cls='match')
mismatch_average = my_epoch.average_signals(cls='mismatch')

plt.plot(match_average, label='match')
plt.plot(mismatch_average, label='mismatch')
plt.legend()
plt.show()
