brojevi = []

while True:
    unos = input("")

    if unos == 'Done' : 
        break
    
    try:
        broj = float(unos)
        brojevi.append(broj)
    except ValueError:
        print("nisi unio broj")
    
BrojElemenata = len(brojevi)


SrednjaVrijednost = sum(brojevi)/len(brojevi)


MinimalnaVrijednost = min(brojevi)
MaksimalnaVrijednost = max(brojevi) 

brojevi.sort()

print(f"Količina unesenih brojeva: {BrojElemenata}")
print(f"Srednja vrijednost: {SrednjaVrijednost:.2f}")
print(f"Minimalna vrijednost: {MinimalnaVrijednost}")
print(f"Maksimalna vrijednost: {MaksimalnaVrijednost}")
print(f"Sortirana lista: {brojevi}")
