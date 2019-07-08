import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--infolder', type=str)
parser.add_argument('--outfile', type=str)


args = parser.parse_args()

if args.infolder is None or args.outfile is None:
    exit(2)


output = open(args.outfile, "w+")

def chomp(x):
    if x.endswith("\r\n"):
        return x[:-2]
    if x.endswith("\n") or x.endswith("\r"):
        return x[:-1]
    return x


def process(lines):
    person1 = ""
    joined_line = ""
    person = ""
    one_set = {}
    stopwords = ['?', '!', ',', '', "Phua:", "Guang:", "Chong:"]

    for line in lines:
        mystring = line
        mystring = mystring.encode('ascii', 'ignore').decode('ascii')
        # mystring = chomp(mystring)

        start = mystring.find('[')
        end = mystring.find(']')
        processed = []
        if start != -1 and end != -1:
            if "Messages to this chat and calls are now secured with end-to-end encryption." in line:
                continue
            if "image omitted" in line:
                continue

            result = mystring[end + 2:len(mystring) - 1]
            stripped = result.split(' ')

            if stripped[0] != person.lower():
                if "wilson" in person.lower():
                    one_set["wilson"] = joined_line
                    if person1 in one_set and "wilson" in one_set:
                        if one_set[person1] != '' and one_set["wilson"] != '':
                            output.write(one_set[person1] + "\t" + one_set["wilson"] + "\n")
                    # output.write(str(one_set) + "\n")
                    one_set = {}
                else:
                    person1 = person.lower()
                    one_set[person1] = joined_line

                joined_line = ""
                person = stripped[0]

            while len(stripped) > 0 and len(stripped[0]) > 0 and stripped[0][-1] != ':':
                stripped.pop(0)
            if len(stripped) > 0:
                stripped.pop(0)

            for item in stripped:
                item = item.strip()
                if "http" in item:
                    pass
                elif item in stopwords:
                    pass
                else:
                    processed.append(item)
            if (len(processed) == 0):
                pass
            else:
                joined_line += " " + " ".join(processed)
    print(person1)


for filename in os.listdir(args.infolder):
    f = open(args.infolder + filename, "r")
    lines = f.readlines()
    process(lines)
