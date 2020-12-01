out = ''
with open('bundle2.txt') as f:
    for line in f:
        line = line.replace(" ", ',')
        out = out + line
with open('../data/democracy_index.csv', 'a') as target:
    target.write(out)
