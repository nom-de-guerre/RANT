import string
import re as re

def NLM_NormalizeText (text):

    text = SubstCharacters (text)
    text = CleanText (text)

    return text

def CleanText (documentRaw):
    
    document = list (documentRaw)
    docLength = len (documentRaw)

    index = 0

    for i in range (docLength):

        x = documentRaw[i]

        if x >= 'a' and x <= 'z':
            document[index] = x
            index += 1
        elif x >= "A" and x <= "Z":
            document[index] = x.lower ()
            index += 1
        elif x >= "0" and x <= "9":
            document[index] = x
            index += 1
        elif x == " ":
            document[index] = x
            index += 1        
        elif x == ".":
            document[index] = x
            index += 1        
            document[index] = "\n"
            index += 1        
        elif x == ",":
            document[index] = " "
            index += 1        
        elif x == "\n":
            document[index] = " "
            index += 1

    document = "".join (document[0:index])
    document = re.sub (r"\d+", " [NUM] ", document)
    document = re.sub (r"\.", " [SEP]. ", document)

    return document

def SubstCharacters (text):

    text = text.replace (",", " ")
    text = text.replace (";", " ")
    text = text.replace ("!", ".")

    text = text.replace ("î", "i")
    text = text.replace ("ï", "i")

    text = text.replace ("œ", "oe")
    text = text.replace ("ô", "o")
    text = text.replace ("ö", "o")
    text = text.replace ("ò", "o")

    text = text.replace ("ü", "u")
    text = text.replace ("û", "u")

    text = text.replace ("α", "a")
    text = text.replace ("ä", "a")
    text = text.replace ("â", "a")
    text = text.replace ("à", "a")
    text = text.replace ("á", "a")
    text = text.replace ("À", "A")

    text = text.replace ("ῥ", "p")

    text = text.replace ("é", "e")
    text = text.replace ("è", "e")
    text = text.replace ("ê", "e")

    text = text.replace ("Ç", "C")
    text = text.replace ("ç", "c")

    text = text.replace ("æ", "ae")

    return text

f = open("../Data/SherlockHolmes.txt", 'r') # 'r' = read
lines = f.read()
f.close()

print (NLM_NormalizeText (lines))

