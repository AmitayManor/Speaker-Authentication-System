
def append_to_file(file, str):
    f = open(file, "a")
    f.write(str + '\n')
    f.close()
