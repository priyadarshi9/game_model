import pandas as pd

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense



df = pd.read_csv("output.json")




columns = [
    "id", 'firstMove',
]


coord_columns = ['snap' + str(i) for i in range(25)]

columns.extend(coord_columns)

columns.extend(['move_x', 'move_y', 'winner'])

df.columns = columns

df_1 = df[df['winner'] == 1]


labels = []
for i in range(len(df_1)):
    if i < 10:
        print(df_1.iloc[i, 27], "  ", df_1.iloc[i, 28] )
    label = 5 * df_1.iloc[i, 27] + df_1.iloc[i, 28]
    labels.append(label)

df_1["label"] = labels

df_1 = df_1.drop(columns = ['id', 'firstMove', 'move_x', 'move_y', 'winner'])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(25,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(25)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

y = df_1.pop('label').values
x = df_1.values


model.fit(x, y, batch_size=256, epochs=100)

model.save("myGameModel.h5")
print("Done.")




