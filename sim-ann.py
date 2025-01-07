import openrouteservice
import random
import math
import os
from dotenv import load_dotenv

# === 1. Definicja miast wraz z ich współrzędnymi i zapotrzebowaniem ===
# Kraków jako depot (indeks 0) z zapotrzebowaniem 0.
cities = [
    ("Kraków", (50.06143, 19.93658), 0),  # Depot
    ("Białystok", (53.13333, 23.15), 500),
    ("Bielsko-Biała", (49.82238, 19.05838), 50),
    ("Chrzanów", (50.13554, 19.40262), 400),
    ("Gdańsk", (54.35205, 18.64637), 200),
    ("Gdynia", (54.51889, 18.53054), 100),
    ("Gliwice", (50.29761, 18.67658), 40),
    ("Gromnik", (49.83562, 20.99735), 200),
    ("Katowice", (50.27065, 19.01543), 300),
    ("Kielce", (50.87033, 20.62752), 30),
    ("Krosno", (49.68863, 21.77019), 60),
    ("Krynica", (49.4216, 20.95731), 50),
    ("Lublin", (51.24645, 22.56844), 60),
    ("Łódź", (51.7592485, 19.4559833), 160),
    ("Malbork", (54.04039, 19.02754), 100),
    ("Nowy Targ", (49.47702, 20.03268), 120),
    ("Olsztyn", (53.77994, 20.49416), 300),
    ("Poznań", (52.40692, 16.92993), 100),
    ("Puławy", (51.41667, 21.96667), 200),
    ("Radom", (51.40253, 21.14714), 100),
    ("Rzeszów", (50.04132, 21.99901), 60),
    ("Sandomierz", (50.68231, 21.74947), 200),
    ("Szczecin", (53.42894, 14.55302), 150),
    ("Szczucin", (50.31552, 21.07419), 60),
    ("Szklarska Poręba", (50.82727, 15.52206), 50),
    ("Tarnów", (50.01381, 20.98698), 70),
    ("Warszawa", (52.22977, 21.01178), 200),
    ("Wieliczka", (49.98718, 20.06477), 90),
    ("Wrocław", (51.10788, 17.03854), 40),
    ("Zakopane", (49.29918, 19.94956), 200),
    ("Zamość", (50.72314, 23.25196), 300)
]

NUM_TRUCKS = 5
TRUCK_CAPACITY = 1000
NUM_CITIES = len(cities)  # 31 (w tym Kraków)

# === 0. Ładowanie zmiennych środowiskowych z pliku .env ===
load_dotenv()  # Wczytuje zmienne z .env
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Brak klucza API! Upewnij się, że plik .env zawiera API_KEY.")

client = openrouteservice.Client(key=API_KEY)

# Przygotowanie listy współrzędnych w formacie (lon, lat) dla wszystkich miast
coords = []
for _, (lat, lon), _ in cities:
    coords.append((lon, lat))

# Pobieramy macierz odległości
response = client.distance_matrix(
    locations=coords,
    profile='driving-car',
    metrics=['distance'],
    validate=False
)

matrix = response['distances']
distance_matrix_ors = {}
for i in range(len(cities)):
    for j in range(len(cities)):
        if i == j:
            distance_matrix_ors[(i, j)] = 0.0
        else:
            distance_matrix_ors[(i, j)] = matrix[i][j] / 1000.0  # zmieniamy na km

# === 3. Zapis macierzy odległości do pliku TXT ===
with open("distances.txt", "w", encoding="utf-8") as f:
    for i in range(len(cities)):
        for j in range(len(cities)):
            if i != j:
                city1 = cities[i][0]
                city2 = cities[j][0]
                km = distance_matrix_ors[(i, j)]
                f.write(f"{city1} {city2} {km:.2f} km\n")

print("Macierz odległości została zapisana do pliku distances.txt.")

# === 4. Funkcje pomocnicze dla algorytmu symulowanego wyżarzania ===
distance_matrix = distance_matrix_ors

def route_distance(route):
    """Oblicza dystans dla jednej trasy, zakładając start i koniec w Krakowie (depot)."""
    if not route:
        return 0
    # Dystans od Krakowa do pierwszego klienta
    total = distance_matrix[(0, route[0])]
    # Dystans między klientami w trasie
    for i in range(len(route) - 1):
        total += distance_matrix[(route[i], route[i + 1])]
    # Dystans z ostatniego klienta z powrotem do Krakowa
    total += distance_matrix[(route[-1], 0)]
    return total


def total_distance(solution):
    """Oblicza całkowity dystans dla wszystkich tras."""
    return sum(route_distance(route) for route in solution)


def is_feasible(solution):
    """Sprawdza, czy żadna trasa nie przekracza pojemności pojazdu."""
    for route in solution:
        load = sum(cities[i][2] for i in route)
        if load > TRUCK_CAPACITY:
            return False
    return True


def generate_initial_solution():
    """Generuje początkowe rozwiązanie VRP, przypisując klientów do samochodów."""
    # Indeksy klientów od 1 do 30 (pomijamy indeks 0 - Kraków)
    indices = list(range(1, NUM_CITIES))
    random.shuffle(indices)

    solution = [[] for _ in range(NUM_TRUCKS)]
    for city_idx in indices:
        placed = False
        for truck_idx in range(NUM_TRUCKS):
            current_load = sum(cities[c][2] for c in solution[truck_idx])
            if current_load + cities[city_idx][2] <= TRUCK_CAPACITY:
                solution[truck_idx].append(city_idx)
                placed = True
                break
        if not placed:
            # Awaryjnie umieszczamy w ostatnim pojeździe, mimo przekroczenia pojemności
            solution[-1].append(city_idx)
    return solution


def generate_neighbor(solution):
    """Generuje sąsiednie rozwiązanie poprzez zamianę dwóch losowych miast między trasami."""
    new_solution = [r[:] for r in solution]
    t1, t2 = random.sample(range(NUM_TRUCKS), 2)
    if not new_solution[t1] or not new_solution[t2]:
        return new_solution
    i1 = random.randint(0, len(new_solution[t1]) - 1)
    i2 = random.randint(0, len(new_solution[t2]) - 1)
    new_solution[t1][i1], new_solution[t2][i2] = new_solution[t2][i2], new_solution[t1][i1]
    return new_solution


def simulated_annealing(iterations=1000, initial_temp=1000.0, alpha=0.99):
    current_solution = generate_initial_solution()
    # Upewniamy się, że początkowe rozwiązanie jest wykonalne
    while not is_feasible(current_solution):
        current_solution = generate_initial_solution()

    best_solution = current_solution
    best_distance = total_distance(best_solution)
    temperature = initial_temp

    for _ in range(iterations):
        neighbor = generate_neighbor(current_solution)
        if not is_feasible(neighbor):
            continue

        current_dist = total_distance(current_solution)
        neighbor_dist = total_distance(neighbor)

        if neighbor_dist < current_dist:
            current_solution = neighbor
            if neighbor_dist < best_distance:
                best_solution = neighbor
                best_distance = neighbor_dist
        else:
            diff = neighbor_dist - current_dist
            acceptance_prob = math.exp(-diff / temperature)
            if random.random() < acceptance_prob:
                current_solution = neighbor

        temperature *= alpha

    return best_solution, best_distance


# === 5. Uruchomienie algorytmu symulowanego wyżarzania ===
solution, dist = simulated_annealing(iterations=2000, initial_temp=2000.0, alpha=0.995)

print("\nNajlepsze znalezione rozwiązanie (trasy):")
for truck_idx, route in enumerate(solution):
    # Wyświetlamy nazwy miast dla każdej trasy
    route_cities = [cities[idx][0] for idx in route]
    print(f"Samochód {truck_idx + 1}: {route_cities}")

print(f"\nCałkowity dystans: {dist:.2f} km")