

'''
validation_ratio = 0.03
num_train_samples = len(trainset)
num_validation_samples = int(num_train_samples * validation_ratio)
indices = torch.randperm(num_train_samples).tolist()
validation_indices = indices[:num_validation_samples]
train_indices = indices[num_validation_samples:]

validation_set = Subset(trainset, validation_indices)
train_set = Subset(trainset, train_indices)

traindata = DataLoader(train_set, batch_size=128, shuffle=True)
valdata = DataLoader(validation_set, batch_size=128, shuffle=False)
'''
