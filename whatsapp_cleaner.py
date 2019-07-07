import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str)
parser.add_argument('--outfile', type=str)
parser.add_argument('--person1', type=str)
parser.add_argument('--person2', type=str)


args = parser.parse_args()

if args.infile is None or args.outfile is None:
    exit(2)

f = open(args.infile, "r")
lines = f.readlines()

output = open(args.outfile, "w+")

def chomp(x):
    if x.endswith("\r\n"):
        return x[:-2]
    if x.endswith("\n") or x.endswith("\r"):
        return x[:-1]
    return x


joined_line = ""
person = ""
one_set = {}
for line in lines:
    mystring = line
    mystring = mystring.encode('ascii', 'ignore').decode('ascii')
    # mystring = chomp(mystring)

    start = mystring.find('[')
    end = mystring.find(']')
    processed = []
    if start != -1 and end != -1:
        result = mystring[end + 2:len(mystring) - 1]
        stripped = result.split(' ')
        if "video" and "omitted" in stripped:
            continue

        if stripped[0] != person:
            if args.person1 in person.lower():
                one_set[args.person1] = joined_line
            elif "wilson" in person.lower():
                one_set["wilson"] = joined_line
                if args.person1 in one_set and "wilson" in one_set:
                    output.write(one_set[args.person1] + "\t" + one_set["wilson"] + "\n")
                    pass
                # output.write(str(one_set) + "\n")
                one_set = {}

            joined_line = ""
            person = stripped[0]

        stripped.pop(0)

        for item in stripped:
            item = item.strip()
            if "http" in item:
                pass
            elif item in ['?', '!', ',', '', "Phua:", "Guang:", "Chong:"]:
                pass
            else:
                processed.append(item)
        if (len(processed) == 0):
            pass
        else:
            joined_line += " " + " ".join(processed)
