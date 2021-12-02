import tensorflow as tf
import numpy as np

def createDetailedPredictions(preds):
	print("Probabilities of next moves are in the decreasing order:")
	preds = preds[0]
	preds_with_pos = [(preds[i], i) for i in range(25)]
	preds_with_pos.sort(reverse=True)

	for i in range(25):
		next_x = preds_with_pos[i][1] // 5
		next_y = preds_with_pos[i][1] % 5
		print("Next Move: ", f'({next_x} , {next_y})', " | Probability: ", preds_with_pos[i][0])
	print("\n", "="*30, "\n")

model = tf.keras.models.load_model('./player_1_moves_model.h5')

flag = 'y'
while flag in 'yY':
	inp = input("Please input the flattened state separated by space (Press n to terminate):")

	if inp in 'nN':
		break
	inp = inp.split(" ")
	int_input = np.array([int(i) for i in inp])
	int_input = int_input.reshape(1,25)

	preds = model.predict(int_input)
	pred_pos = np.argmax(preds)

	next_x = pred_pos//5
	next_y = pred_pos % 5
	print("Your next move should be: ", f'{next_x}, {next_y}', '\n' )

	det = input("If you want the detailed prediction probabilities, enter y, else press enter: ")
	if det == "":
		continue
	elif det in 'yY':
		createDetailedPredictions(preds)


print("Program Terminated...")
print("Bye.")