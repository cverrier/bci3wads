from curses import window
from bci3wads.features import processing
from bci3wads.utils import constants

subject = processing.Subject('Subject_B_Train.pickle')
channel_ids = [11]

for epoch_id in range(len(subject.epochs)):
    epoch = subject.epochs[epoch_id]

    processed_epoch_channels = subject.process_epoch_channels(
        epoch_id=epoch_id,
        channel_ids=channel_ids,
        window_size=constants.WINDOW_SIZE
    )

    data = subject.process_epoch(
        processed_epoch_channels,
        target_char=epoch.target_char,
        target_char_codes=epoch.target_char_codes(),
        target_char_coords=epoch.target_char_coords(),
        epoch_id=epoch_id,
        channel_ids=channel_ids
    )

    subject.save_epoch(data)
