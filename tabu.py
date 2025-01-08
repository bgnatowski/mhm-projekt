import os
from dotenv import load_dotenv
import openrouteservice
import random
import math
from collections import deque

# === 0. Ładowanie zmiennych środowiskowych z pliku .env ===
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Brak klucza API! Upewnij się, że plik .env zawiera API_KEY.")

# === 1. Definicja miast (Kraków + 30 klientów) ===
# Format: (nazwa_miasta, (latitude, longitude), zapotrzebowanie)
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
NUM_CITIES = len(cities)  # 31

# === 2. Pobieranie macierzy odległości z OpenRouteService ===
client = openrouteservice.Client(key=API_KEY)

coords = [(lon, lat) for (_, (lat, lon), _) in cities]  # (lon, lat)

response = client.distance_matrix(
    locations=coords,
    profile='driving-car',
    metrics=['distance'],
    validate=False
)

matrix = response['distances']
distance_matrix_ors = {}
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        if i == j:
            distance_matrix_ors[(i, j)] = 0.0
        else:
            distance_matrix_ors[(i, j)] = matrix[i][j] / 1000.0  # w km

# === 3. Zapis macierzy do pliku TXT ===
with open("distances.txt", "w", encoding="utf-8") as f:
    for i in range(NUM_CITIES):
        for j in range(NUM_CITIES):
            if i != j:
                city1 = cities[i][0]
                city2 = cities[j][0]
                km = distance_matrix_ors[(i, j)]
                f.write(f"{city1} {city2} {km:.2f}\n")

print("Macierz odległości zapisana do pliku distances.txt.")

# === 4. Funkcje pomocnicze (koszt trasy, feasibility, itd.) ===
distance_matrix = distance_matrix_ors


def route_distance(route):
    """Oblicza dystans dla jednej trasy (start i koniec w Krakowie = index 0)."""
    if not route:
        return 0.0
    dist = distance_matrix[(0, route[0])]  # Kraków -> pierwszy klient
    for i in range(len(route) - 1):
        dist += distance_matrix[(route[i], route[i + 1])]
    dist += distance_matrix[(route[-1], 0)]  # ostatni klient -> Kraków
    return dist


def total_distance(solution):
    """Suma dystansów wszystkich tras."""
    return sum(route_distance(r) for r in solution)


def is_feasible(solution):
    """Sprawdza, czy w żadnym pojeździe nie przekroczono pojemności."""
    for route in solution:
        load = sum(cities[idx][2] for idx in route)
        if load > TRUCK_CAPACITY:
            return False
    return True


def generate_initial_solution():
    """Rozkłada klientów (1..30) losowo na 5 pojazdów."""
    clients = list(range(1, NUM_CITIES))
    random.shuffle(clients)
    solution = [[] for _ in range(NUM_TRUCKS)]
    for city_idx in clients:
        placed = False
        for truck_idx in range(NUM_TRUCKS):
            current_load = sum(cities[c][2] for c in solution[truck_idx])
            if current_load + cities[city_idx][2] <= TRUCK_CAPACITY:
                solution[truck_idx].append(city_idx)
                placed = True
                break
        if not placed:
            # Awaryjnie wrzucamy do ostatniego, nawet jeśli przekroczy
            solution[-1].append(city_idx)
    return solution


# === 4A. Lokalny 2-opt w obrębie jednej trasy ===
def two_opt(route):
    """
    Wykonuje prostą lokalną optymalizację 2-opt dla pojedynczej listy klientów.
    Trasa jest interpretowana: Kraków -> route[0] -> ... -> route[n-1] -> Kraków
    Zwraca (ew. ulepszoną) trasę.
    """
    if len(route) < 2:
        return route

    best_route = route[:]
    best_dist = route_distance(best_route)
    improved = True

    while improved:
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:]
                # odwracamy segment [i, j]
                new_route[i:j + 1] = reversed(new_route[i:j + 1])
                new_dist = route_distance(new_route)
                if new_dist < best_dist:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return best_route


def local_search_2opt(solution):
    """
    Wykonuje 2-opt dla każdej trasy w solution i zwraca ulepszoną kopię.
    """
    new_sol = []
    for route in solution:
        new_r = two_opt(route)
        new_sol.append(new_r)
    return new_sol


# === 5. Definiujemy rozszerzone sąsiedztwo i ruchy Tabu ===

def swap_cities(solution):
    """Losowa zamiana miast między dwoma pojazdami."""
    new_sol = [r[:] for r in solution]  # kopiujemy
    t1, t2 = random.sample(range(NUM_TRUCKS), 2)
    if not new_sol[t1] or not new_sol[t2]:
        return new_sol, None

    i1 = random.randint(0, len(new_sol[t1]) - 1)
    i2 = random.randint(0, len(new_sol[t2]) - 1)
    cityA, cityB = new_sol[t1][i1], new_sol[t2][i2]

    new_sol[t1][i1], new_sol[t2][i2] = cityB, cityA
    move = ("swap", t1, i1, t2, i2, cityA, cityB)
    return new_sol, move


def move_city(solution):
    """Losowe przeniesienie jednego miasta z trasy t1 do trasy t2."""
    new_sol = [r[:] for r in solution]
    # wybieramy losowo trasę źródłową, ale tylko taką, która nie jest pusta
    non_empty_trucks = [k for k in range(NUM_TRUCKS) if new_sol[k]]
    if not non_empty_trucks:
        return new_sol, None
    t1 = random.choice(non_empty_trucks)
    t2 = random.choice(range(NUM_TRUCKS))
    if t1 == t2:
        return new_sol, None

    i1 = random.randint(0, len(new_sol[t1]) - 1)
    cityA = new_sol[t1].pop(i1)
    new_sol[t2].append(cityA)
    move = ("move", t1, i1, t2, len(new_sol[t2]) - 1, cityA, None)
    return new_sol, move


def swap_within_route(solution):
    """
    Prosta zamiana dwóch klientów w obrębie tej samej trasy.
    """
    new_sol = [r[:] for r in solution]
    t = random.choice([k for k in range(NUM_TRUCKS) if len(new_sol[k]) > 1])
    route_len = len(new_sol[t])
    i1, i2 = random.sample(range(route_len), 2)
    new_sol[t][i1], new_sol[t][i2] = new_sol[t][i2], new_sol[t][i1]
    move = ("swap_in", t, i1, i2, None, None, None)
    return new_sol, move


def generate_neighbors(solution, k=10):
    """
    Generuje zbiór sąsiednich rozwiązań przez różne typy ruchów.
    Losowo wybieramy jeden z operatorów: swap_cities, move_city, swap_within_route.
    Dla każdego sąsiada wykonujemy też local search (2-opt) i dopiero wtedy zwracamy.
    """
    neighbors = []
    operators = [swap_cities, move_city, swap_within_route]
    for _ in range(k):
        op = random.choice(operators)
        nsol, move = op(solution)
        if move is not None:
            # Dodajemy local search 2-opt
            nsol_ls = local_search_2opt(nsol)
            neighbors.append((nsol_ls, move))
        else:
            # jeśli move=None, to po prostu wstawiamy oryginał
            neighbors.append((solution, None))
    return neighbors


# === 6. Implementacja Tabu Search z local search i rozszerzonym sąsiedztwem ===

def tabu_search(max_iters=700, tabu_tenure=15, neighbor_size=40):
    """
    - max_iters: maksymalna liczba iteracji
    - tabu_tenure: ile iteracji dany ruch jest tabu
    - neighbor_size: ile sąsiadów generujemy w każdej iteracji

    Dodajemy:
    - local_search_2opt do każdej generacji sąsiada
    - operator swap_within_route w generate_neighbors
    - dłuższy max_iters i większy neighbor_size, by mocniej przeszukać
    """
    # 1. Generujemy rozwiązanie początkowe
    current_solution = generate_initial_solution()
    while not is_feasible(current_solution):
        current_solution = generate_initial_solution()

    # 2. Inicjujemy best_solution
    best_solution = current_solution
    best_cost = total_distance(best_solution)

    # 3. Lista (słownik) Tabu
    tabu_list = {}

    iteration = 0
    while iteration < max_iters:
        iteration += 1
        neighbors = generate_neighbors(current_solution, k=neighbor_size)

        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None

        for (nsol, move) in neighbors:
            if not is_feasible(nsol):
                continue
            cost_nsol = total_distance(nsol)

            # sprawdzamy, czy ruch jest tabu
            if move in tabu_list and tabu_list[move] > iteration:
                # ruch jest tabu, ale sprawdzamy aspirację
                if cost_nsol < best_cost:
                    # łamiemy tabu
                    if cost_nsol < best_neighbor_cost:
                        best_neighbor = nsol
                        best_neighbor_cost = cost_nsol
                        best_move = move
                else:
                    # pomijamy
                    continue
            else:
                # nie jest tabu
                if cost_nsol < best_neighbor_cost:
                    best_neighbor = nsol
                    best_neighbor_cost = cost_nsol
                    best_move = move

        # Jeśli nie znaleźliśmy żadnego feasible sąsiada spoza tabu
        if best_neighbor is None:
            print("Brak możliwych sąsiadów w tej iteracji - przerwanie.")
            break

        # 4. Aktualizujemy bieżące rozwiązanie
        current_solution = best_neighbor
        current_cost = best_neighbor_cost

        # 5. Dodajemy ruch do tabu
        if best_move:
            tabu_list[best_move] = iteration + tabu_tenure

        # 6. Sprawdzamy, czy mamy nowe global best
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost

        # 7. Usuwamy z tabu ruchy, które wygasły
        moves_to_remove = []
        for mv, expiry in tabu_list.items():
            if expiry <= iteration:
                moves_to_remove.append(mv)
        for mv in moves_to_remove:
            del tabu_list[mv]

        # Co 10 iteracji wyświetlamy postęp
        if iteration % 10 == 0:
            print(f"Iteracja {iteration}, najlepszy koszt globalny: {best_cost:.2f}")

    return best_solution, best_cost


# === 7. Uruchomienie Tabu Search i wyświetlenie wyników ===
if __name__ == "__main__":
    # Możesz dostosować parametry w zależności od możliwości czasowych
    solution, dist = tabu_search(max_iters=500, tabu_tenure=8, neighbor_size=110)

    print("\n=== Najlepsze rozwiązanie znalezione przez Tabu Search (z 2-opt) ===")
    for truck_idx, route in enumerate(solution):
        route_names = [cities[r][0] for r in route]
        load = sum(cities[r][2] for r in route)
        print(f"Samochód {truck_idx + 1}: {route_names} | ładunek={load}")
    print(f"\nCałkowity dystans: {dist:.2f} km")