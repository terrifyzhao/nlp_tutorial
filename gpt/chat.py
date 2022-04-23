from gpt.chitchat.interact import chitchat

while 1:
    text = input('text:')
    r = chitchat(text)
    print(r)
