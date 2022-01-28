from bci3wads.features import preprocessing

subject = preprocessing.Subject('Subject_B_Train.mat')
data = subject.clean_data()
subject.save(data)
