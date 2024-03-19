
#%%
import pickle
import jax.numpy as np

#%% DATA u train data

# 빈 데이터셋 생성
merged_dataset = np.empty((0, 100))
names = ['1', '0_001', '0_1', '0_01']

# 파일 경로 리스트 구성
file_paths = ['../data/data_u_train_unif_{}.pkl'.format(i) for i in names]
# 파일 경로 리스트 순회하면서 데이터셋 병합
for file_path in file_paths:
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
        merged_dataset = np.vstack((merged_dataset, dataset))

# 병합된 데이터셋의 크기 출력
print("Merged Dataset Shape:", merged_dataset.shape)

output_file_path = '../data/u_mix_unif_001_1_8k.pkl'
with open(output_file_path, 'wb') as output_file:
    pickle.dump(merged_dataset, output_file)

# %% DATA s train data

# 빈 데이터셋 생성
merged_dataset = np.empty((0, 3))

# 파일 경로 리스트 구성
file_paths = ['../data/data_s_train_unif_{}.pkl'.format(i) for i in names]

# 파일 경로 리스트 순회하면서 데이터셋 병합
for file_path in file_paths:
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
        merged_dataset = np.vstack((merged_dataset, dataset))

# 병합된 데이터셋의 크기 출력
print("Merged Dataset Shape:", merged_dataset.shape)

output_file_path = '../data/s_mix_unif_001_1_8k.pkl'
with open(output_file_path, 'wb') as output_file:
    pickle.dump(merged_dataset, output_file)
#%% DATA y train data

# 빈 데이터셋 생성
merged_dataset = np.empty((0, 1))

# 파일 경로 리스트 구성
file_paths = ['../data/data_y_train_unif_{}.pkl'.format(i) for i in names]

# 파일 경로 리스트 순회하면서 데이터셋 병합
for file_path in file_paths:
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
        merged_dataset = np.vstack((merged_dataset, dataset))

# 병합된 데이터셋의 크기 출력
print("Merged Dataset Shape:", merged_dataset.shape)

output_file_path = '../data/y_mix_unif_001_1_8k.pkl'
with open(output_file_path, 'wb') as output_file:
    pickle.dump(merged_dataset, output_file)
# %%
