from bci3wads.features import preprocessing

subject = preprocessing.Subject('Subject_A_Train.pickle', is_train=True)
data = subject.clean_data()
subject.save(data)
