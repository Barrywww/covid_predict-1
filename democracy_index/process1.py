out = ''
with open('bundle1.txt') as f:
    i = 1
    curr_line = ''
    for line in f:
        line = line.strip()
        if i % 8 != 0:
            curr_line = curr_line + line + ','
        else:
            curr_line = curr_line + line + '\n'
            out = out + curr_line
            curr_line = ''
        i += 1
with open('../data/democracy_index.csv', 'w') as file:
    file.write(out)
