import pandas as pd

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("TRAIN FIRST 5 ROWS")
print(train_df.head())

print("\nTRAIN COLUMNS")
print(train_df.columns)

print("\nTRAIN SHAPE")
print(train_df.shape)

print("\nTRAIN MISSING VALUES")
print(train_df.isnull().sum())

print("\n" + "="*50)

print("TEST FIRST 5 ROWS")
print(test_df.head())

print("\nTEST COLUMNS")
print(test_df.columns)

print("\nTEST SHAPE")
print(test_df.shape)

print("\nTEST MISSING VALUES")
print(test_df.isnull().sum())