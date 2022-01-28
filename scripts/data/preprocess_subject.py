from bci3wads.features import preprocessing

subject = preprocessing.Subject('Subject_B_Train.mat')
subject.clean_data()
subject.save()
