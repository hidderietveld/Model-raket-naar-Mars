import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Constants
G = 6.67430e-11 # Gravitatieconstante (m^3 kg^-1 s^-2)
dt = 86400# Time step (6 uur)
t_eind = 302 # Eindtijd (dagen)        Dit is de tijd tot het dichtsbijzijnde punt, dit kan je makkelijk aanpassen
extra_v = 676.6


# Celestial bodies class
class Hemellichaam:
    def __init__(self, naam, m, beginx, beginv, kleur):
        self.naam = naam 
        self.m = m # massa
        self.x = np.array(beginx, dtype=float)          # arrays worden hier gebruikt als vectoren 
        self.v = np.array(beginv, dtype=float)          # dtype=float geeft het datatype aan, in dit geval een float (kommagetal)
        self.F = np.zeros(2)                            # maakt een lege array aan [0.0, 0.0], je begint met een kracht van 0 
        self.baan = [self.x.copy()]                     # slaat alle posities op voor elke dt, om zo een plot te kunnen maken
        self.kleur = kleur                              # kleur van het object in matplotlib
   
    def bereken_F(self, other):
        x = other.x - self.x        # Relatieve positie, vector van zichzelf naar ander object
        d = np.linalg.norm(x)       # De lengte van deze vector, en dus de afstand tussen de twee objecten. np.linalg.norm(x) is hetzelfde als de absolute waarde van de vector
        if d < 1e-10:
            return np.zeros(2)
        richting = x / d                                # Richtingsvector met lengte 1, wordt gebruikt voor vermenigvuldigingen
        F_g_waarde = G * self.m * other.m / (d ** 2)    # Dit is alleen de lengte van de vector. Gravitatiekracht formule.
        F_g = F_g_waarde * richting                     # Gravitatiekracht vector, door het te vermenigvuldigen met de vector krijg je de daadwerkelijke vector
        return F_g
    
    # we hebben gekozen eee
    def update_x(self):
        self.x += self.v * dt               # x = x + v*dt
        self.baan.append(self.x.copy())     # voegt de nieuwste positiewaarde toe aan de lijst van alle waardes

    def update_v(self):
        a = self.F / self.m         # a = F/m  Tweede wet van Newton
        self.v += a * dt            # v = v + a*dt

#Het aanmaken van de verschillende hemellichamen, gegevens van NASA

zon = Hemellichaam("Zon", 1.989e30, [0, 0], [0, 0], "yellow")
aarde = Hemellichaam("Aarde", 5.972e24, [1.496e11, 0], [0, 29.78e3], "blue")
mars = Hemellichaam("Mars", 1e2, [1.74125490e11, 1.45551037e11], [-15545.6, 18509.0], "red")
#mars = Hemellichaam("Mars", 6.39e23, [2.279e11, 0], [0, 24.07e3], "red")


r_aarde = np.linalg.norm(aarde.x)   # Baanstraal aarde
r_mars = np.linalg.norm(mars.x)     # Baanstraal mars

raket_v = aarde.v.copy()    # Snelheid raket = snelheid aarde
raket_x = aarde.x.copy()    # Positie raket = positie aarde
raket_m = 3000000           # massa raket
raket_c = "black"           # kleur raket
semi_major_axis = (r_aarde + r_mars) / 2                                # Semi-major-axis, zoals uitgelegd in H6
v_perigee = np.sqrt(G * zon.m * (2 / r_aarde - 1 / semi_major_axis))    # Vis-viva formule
dv = v_perigee - np.linalg.norm(aarde.v) + extra_v                      # Dv voor transferbaan benodigdheden
raket_v[1] += dv

raket = Hemellichaam("Raket", raket_m, raket_x, raket_v, raket_c)       # Aanmaken van de raket

lichamen = [zon, aarde, mars, raket] # lijst met alle lichamen, voor for-loops

fig, ax = plt.subplots()            # zodat dit niet steeds opnieuw getypt hoeft te worden
plt.subplots_adjust(bottom=0.25)    # plek voor de slider
ax.set_aspect('equal')              # view blijft altijd een vierkant, zo kunnen de resultaten niet onrealistisch worden
ax.set_facecolor('gray')            # achtergrond kleur
ax.grid(True)                       # grid

scatter_plots = [] # lijst voor bewaren van gegevens om later te plotten
for lichaam in lichamen:
    scatter = ax.scatter(lichaam.x[0], lichaam.x[1], s=100 if lichaam.naam != "Raket" else 20, c=lichaam.kleur, label=lichaam.naam) #lichaam.x[0] is de x-waarde van pos en lichaam.x[1] is de y-waarde
    scatter_plots.append(scatter) # voegt deze gegevens toe aan de lijst
 
lijnen = [] # lijst voor bewaren van gegevens om later te plotten
for lichaam in lichamen:
    lijn, = ax.plot([], [], c=lichaam.kleur, alpha = 0.5) # alpha is de opacity, x en y waardes zijn leeggelaten omdat dat later pas toegevoegd wordt
    lijnen.append(lijn) # voegt deze gegevens toe aan de lijst

grootste_uitwijking = max(np.linalg.norm(lichaam.x) for lichaam in lichamen) # grootste afstand van de zon
ax.set_xlim(-grootste_uitwijking * 1.2, grootste_uitwijking * 1.2) # dit wordt hier gebruikt om de grootte automatisch aan te passen
ax.set_ylim(-grootste_uitwijking * 1.2, grootste_uitwijking * 1.2)
ax.legend()
"""
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
"""
ax_time = plt.axes([0.25, 0.1, 0.65, 0.03]) # [linksboven, linksonder, breedte, hoogte] van de Slider
tijd_slider = Slider(ax_time, 'Tijd (dagen)', 0, t_eind-1, valinit=0, valstep=1) # aanmaken van de slider voor de tijd, valinit is de waarde waarmee we beginnen, valstep de stap per berekening en t_eind is wanneer het moet stoppen

tijd_data = []      # Lijst met de positie per dt(dag)
d_mars_raket = []   # Lijst met de afstanden van de raket tot Mars

def bereken_stap(lichamen_lokaal):
    for lichaam in lichamen_lokaal:
        lichaam.F = np.zeros(2) # reset de kracht
    
    for i, lichaam1 in enumerate(lichamen_lokaal):
        for lichaam2 in lichamen_lokaal[i+1:]: # [i+1:] maakt een sublist excl. de index van lichaam 1 (i). Je hoeft niet de zwaarte kracht van lichaam 2 op 1 te meten als je 1 op 2 al hebt berekent
            F = lichaam1.bereken_F(lichaam2) #roept de functie aan voor het berekenen van de zwaartekracht tussen 1 en 2
            lichaam1.F += F # actie = -reactie
            lichaam2.F -= F # je telt de kracht bij de oude kracht op , klinkt gek maar dat doen we omdat we later ook wrijvingskracht moeten meerekenen

    for lichaam in lichamen_lokaal:
        lichaam.update_v() # update de snelheid
        lichaam.update_x() # update de positie

    return lichamen_lokaal



lichamen_kopie = [Hemellichaam(l.naam, l.m, l.x.copy(), l.v.copy(), l.kleur) for l in lichamen] # lijst met alle gegevens van de lichamen
for _ in range(int(t_eind)):
    lichamen_kopie = bereken_stap(lichamen_kopie) # callt de functie

    # Voegt de nieuwe x toe. !!!! Wanneer je ook de snelheid op een bepaald punt wil weten, vervang (l.x.copy()) dan met (l.x.copy(), l.v.copy()). De tracers van de planeten werken dan alleen niet, dus voor mooie plaatjes weer weghalen.
    tijd_data.append([(l.x.copy()) for l in lichamen_kopie]) 

    mars_index = lichamen_kopie.index(
    next(l for l in lichamen_kopie if l.naam == "Mars")                 # De index die Mars heeft in de lichamen_kopie lijst, deze code is er voor als je nieuwe objecten toevoegd, dat hij dit dan nogsteeds voor Mars berekent
    ) 
    raket_index = lichamen_kopie.index(
        next(l for l in lichamen_kopie if l.naam == "Raket")            # Hetzelfde, maar dan voor de Raket
    )
    d = np.linalg.norm(
        lichamen_kopie[mars_index].x - lichamen_kopie[raket_index].x    # Berekent de afstand tussen de raket en Mars    
    )
    d_mars_raket.append(d) # Voegt de afstand toe aan de lijst.


def update(val):
    dag = int(tijd_slider.val) # de positie van de slider

    
    for i, lichaam in enumerate(lichamen): # enumerate geeft elk item in de lijst een index. [0, Aarde, 1, Zon....] Dit doen we omdat we zowel de naam als de index nodig hebben
        scatter_plots[i].set_offsets(tijd_data[dag][i]) # de posities worden geupdate voor de correcte dag. Je gebruikt twee brackets omdat in elke array zit nog een array
    
    for i, lichaam in enumerate(lichamen):
        baan_array = np.array(tijd_data[:dag+1]) # array met de posities van het begin tot de huidige dag. +1 aangezien het eerste getal index0 heeft
        lijnen[i].set_data(baan_array[:, i, 0], baan_array[:, i, 1]) # de data van de lijnen wordt geupdated met de posities van de array. Dit zijn de [] die we in eerste instantie open hebben gelaten
    
    fig.canvas.draw_idle()

tijd_slider.on_changed(update)

plt.show()          # Haal weg voor optimalisatie bij berekeningen

'''
Ik heb hieronder de code neergezet die ik tijdelijk heb gebruikt voor bepaalde gegevens in het onderzoek. 


checkDag = 76 # dagen voor mars tot snijpunt
mars_index = lichamen.index(mars)
mars_x, mars_v = tijd_data[checkDag][mars_index]

print(f"Mars' positie op dag {checkDag}: {mars_x}")
print(f"Mars' velocity op dag {checkDag}: {mars_v}")

#Dit heb ik gebruikt om de hoek te vinden tussen de aarde en de mars, die ik dan weer kan gebruiken om te vinden wanneer die verhouding zo is. Dit doe ik door de gevonden positie van mars in te vullen en dan de formule uit te rekenen.
aarde_x = aarde.x.copy()
hoek_aarde_mars = np.arccos(np.dot(aarde_x, mars_x) / (np.linalg.norm(aarde_x) * np.linalg.norm(mars_x)))
print(f"Hoek tussen de mars en de aarde in graden {np.rad2deg(hoek_aarde_mars}."))

Code voor beste lanceer moment zoeken (Zet gegevens mars als zo: Hemellichaam("Mars", 6.39e23, [2.279e11, 0], [0, 24.07e3], "red")) Voor betere performance, haal plt.show() weg.

hoek = 40.5
plusminus = 0.1

launch_window_dagen = None
for dag in range(len(aarde.baan)):
    aarde_x = aarde.baan[dag]
    mars_x = mars.baan[dag]

    hoek_diff_radialen = (
        np.arccos(np.dot(aarde_x, mars_x) / (np.linalg.norm(aarde_x) * np.linalg.norm(mars_x)))
    )
    hoek_diff_graden = np.degrees(hoek_diff_radialen)

    if abs(hoek_diff_graden - hoek) < plusminus:
        launch_window_dagen = dag
        aarde_pos = aarde_x
        mars_pos = mars_x
        break

if launch_window_days is not None:
    print(f"Launch window (dagen): {launch_window_dagen")
    print(f"Aarde : {aarde_pos}")
    print(f"Mars : {mars_pos}")
else:
    print("Niet mogelijk")

# Tekst voor het uitprinten van de minimale afstand tussen de raket en Mars
print(f"Bij het punt waar de raket en mars het dichts bij elkaar zijn hebben ze een afstand van elkaar van {min(d_mars_raket)/1000}km")
print(f"Dit punt wordt bereikt na {d_mars_raket.index(min(d_mars_raket))} dagen")    
    
'''





