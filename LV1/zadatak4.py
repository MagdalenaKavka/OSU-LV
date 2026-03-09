import string


with open('song.txt', 'r') as file:
    tekst = file.read()

rijeci = tekst.lower().split()

broj_rijeci = {}
brojac = 0

for rijec in rijeci:
    rijec = rijec.strip(string.punctuation)
    if rijec in broj_rijeci:
        broj_rijeci[rijec] += 1
    else:
        broj_rijeci[rijec] = 1  

for rijec, brojac in broj_rijeci.items():
    print(f"{rijec}: {brojac}")

for rijec in broj_rijeci:
    if broj_rijeci[rijec] == 1:
        brojac += 1
    
print(f'Broj različitih riječi koje se pojavljuju samo jednom: {brojac}')

