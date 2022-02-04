from bci3wads.features import preprocessing

subject = preprocessing.Subject('Subject_B_Test.mat', is_train=False)
data = subject.clean_data()
subject.save(data)
