import os
import math
import random
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import openrouteservice
from collections import deque

# ------------------------------------------------------------------------------
#  1. Ładowanie klucza API z pliku .env oraz definicja głównych parametrów
# ------------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Brak klucza API w .env (API_KEY)! Upewnij się, że plik istnieje.")

# Parametry problemu
NUM_TRUCKS = 5
TRUCK_CAPACITY = 1000

# Parametry Tabu Search
MAX_ITERS = 500  # liczba iteracji
TABU_TENURE = 10  # czas „zakazania” ruchu
NEIGHBOR_SIZE = 50  # liczba sąsiadów generowanych w każdej iteracji

# ------------------------------------------------------------------------------
#  2. Dane klientów i Krakowa (zapotrzebowania, współrzędne)
# ------------------------------------------------------------------------------
cities = [
    # (nazwa_miasta, (latitude, longitude), zapotrzebowanie)
    ("Kraków", (50.06143, 19.93658), 0),  # Depot (index 0)
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

NUM_CITIES = len(cities)  # 31

# ------------------------------------------------------------------------------
#  3. Pobranie macierzy odległości z OpenRouteService
# ------------------------------------------------------------------------------
client = openrouteservice.Client(key=API_KEY)

# Tworzymy listę (lon, lat)
coords = [(lon, lat) for (_, (lat, lon), _) in cities]

resp = client.distance_matrix(
    locations=coords,
    profile='driving-car',
    metrics=['distance'],
    validate=False
)

dist_matrix_meters = resp['distances']
distance_matrix = {}
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        if i == j:
            distance_matrix[(i, j)] = 0.0
        else:
            distance_matrix[(i, j)] = dist_matrix_meters[i][j] / 1000.0  # metry -> km


# ------------------------------------------------------------------------------
#  4. Funkcje obliczania dystansu tras i weryfikacji ograniczeń
# ------------------------------------------------------------------------------
def route_distance(route):
    """
    Oblicza dystans dla jednej trasy:
      - Start w Krakowie (index 0),
      - Odwiedza miasta z 'route',
      - Powrót do Krakowa na końcu.
    """
    if not route:
        return 0.0
    dist = distance_matrix[(0, route[0])]  # 0 -> pierwszy
    for i in range(len(route) - 1):
        dist += distance_matrix[(route[i], route[i + 1])]
    dist += distance_matrix[(route[-1], 0)]  # ostatni -> 0
    return dist


def total_distance(solution):
    """Sumuje dystanse wszystkich tras w solution."""
    return sum(route_distance(r) for r in solution)


def is_feasible(solution):
    """Sprawdza, czy żadna trasa nie przekracza pojemności."""
    for route in solution:
        load = sum(cities[idx][2] for idx in route)
        if load > TRUCK_CAPACITY:
            return False
    return True


# ------------------------------------------------------------------------------
#  5. Generowanie rozwiązania początkowego
# ------------------------------------------------------------------------------
def generate_initial_solution():
    """
    Rozdziela klientów (1..30) między 5 pojazdów w sposób losowy,
    zachowując constraints (jeśli się uda).
    """
    clients = list(range(1, NUM_CITIES))  # klienci to indeksy 1..30
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
            # Awaryjnie wrzucamy do ostatniego (może przekroczyć capacity)
            solution[-1].append(city_idx)

    return solution


# ------------------------------------------------------------------------------
#  6. Lokalny 2-opt w obrębie jednej trasy
# ------------------------------------------------------------------------------
def two_opt(route):
    """
    Wykonuje 2-opt w obrębie listy klientów (bez depo),
    co pozwala skracać trasę.
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
    """Wykonuje 2-opt dla każdej trasy w solution."""
    new_sol = []
    for route in solution:
        new_r = two_opt(route)
        new_sol.append(new_r)
    return new_sol


# ------------------------------------------------------------------------------
#  7. Definiujemy rozszerzone sąsiedztwo i ruchy Tabu
# ------------------------------------------------------------------------------
def swap_cities(solution):
    """Losowa zamiana miast między dwoma różnymi pojazdami."""
    new_sol = [r[:] for r in solution]
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
    """Losowe przeniesienie jednego miasta z trasy t1 do t2."""
    new_sol = [r[:] for r in solution]
    non_empty = [k for k in range(NUM_TRUCKS) if new_sol[k]]
    if not non_empty:
        return new_sol, None
    t1 = random.choice(non_empty)
    t2 = random.choice(range(NUM_TRUCKS))
    if t1 == t2:
        return new_sol, None
    i1 = random.randint(0, len(new_sol[t1]) - 1)
    cityA = new_sol[t1].pop(i1)
    new_sol[t2].append(cityA)
    move = ("move", t1, i1, t2, len(new_sol[t2]) - 1, cityA, None)
    return new_sol, move


def swap_within_route(solution):
    """Zamiana kolejności 2 klientów w obrębie tej samej trasy."""
    new_sol = [r[:] for r in solution]
    # Wybieramy trasę, w której jest co najmniej 2 klientów
    candidates = [i for i in range(NUM_TRUCKS) if len(new_sol[i]) > 1]
    if not candidates:
        return new_sol, None
    t = random.choice(candidates)
    if len(new_sol[t]) < 2:
        return new_sol, None
    i1, i2 = random.sample(range(len(new_sol[t])), 2)
    new_sol[t][i1], new_sol[t][i2] = new_sol[t][i2], new_sol[t][i1]
    move = ("swap_in", t, i1, i2, None, None, None)
    return new_sol, move


def generate_neighbors(solution, k=NEIGHBOR_SIZE):
    """
    Generuje k sąsiadów, każdy za pomocą jednego z 3 operatorów:
      swap_cities, move_city, swap_within_route.
    Dodatkowo wykonujemy local_search_2opt na wygenerowanym sąsiedzie.
    """
    neighbors = []
    operators = [swap_cities, move_city, swap_within_route]
    for _ in range(k):
        op = random.choice(operators)
        nsol, move = op(solution)
        if move is not None:
            # local search 2-opt
            nsol_ls = local_search_2opt(nsol)
            neighbors.append((nsol_ls, move))
        else:
            neighbors.append((solution, None))
    return neighbors


# ------------------------------------------------------------------------------
#  8. Implementacja Tabu Search (z local search, rozszerzonym sąsiedztwem)
# ------------------------------------------------------------------------------
def tabu_search(max_iters=MAX_ITERS, tabu_tenure=TABU_TENURE, neighbor_size=NEIGHBOR_SIZE):
    """
    Główna pętla Tabu Search. Zwraca solution, dist.
    """
    current_solution = generate_initial_solution()
    while not is_feasible(current_solution):
        current_solution = generate_initial_solution()

    best_solution = current_solution
    best_cost = total_distance(best_solution)

    # Słownik: ruch -> do której iteracji ruch jest tabu
    tabu_list = {}

    for iteration in range(1, max_iters + 1):
        neighbors = generate_neighbors(current_solution, k=neighbor_size)

        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None

        for (nsol, move) in neighbors:
            if not is_feasible(nsol):
                continue
            cost_nsol = total_distance(nsol)

            # Ruch tabu i aspiracja
            if move in tabu_list and tabu_list[move] > iteration:
                # Ruch jest tabu, sprawdź aspirację (czy jest lepszy od global best)
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
                # Nie jest tabu
                if cost_nsol < best_neighbor_cost:
                    best_neighbor = nsol
                    best_neighbor_cost = cost_nsol
                    best_move = move

        if best_neighbor is None:
            print("Brak możliwych sąsiadów w tej iteracji (lub wszystkie tabu). Zatrzymanie.")
            break

        # Aktualizujemy bieżące rozwiązanie
        current_solution = best_neighbor
        current_cost = best_neighbor_cost

        # Aktualizacja tabu
        if best_move:
            tabu_list[best_move] = iteration + tabu_tenure

        # Sprawdzenie lepszego globalnie
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost

        # Usuwanie ruchów, których tabu już wygasło
        to_remove = []
        for mv, expiry in tabu_list.items():
            if expiry <= iteration:
                to_remove.append(mv)
        for mv in to_remove:
            del tabu_list[mv]

        # Wypis postępu co 10 iteracji
        if iteration % 10 == 0:
            print(f"Iteracja {iteration}, najlepszy dystans globalny = {best_cost:.2f}")

    return best_solution, best_cost


# ------------------------------------------------------------------------------
#  9. Wizualizacja graficzna rozwiązania za pomocą matplotlib
# ------------------------------------------------------------------------------
def visualize_solution(solution):
    """
    Rysuje miasta na mapie (współrzędne z 'cities'),
    łączy je wg tras solution (różnymi kolorami).
    """
    plt.figure(figsize=(8, 8))
    # Kolory do 5 pojazdów (jeśli chcesz więcej, rozszerz listę)
    colors = ["red", "green", "blue", "orange", "purple"]

    # Rysowanie punktów (miast)
    for i, (name, (lat, lon), demand) in enumerate(cities):
        plt.scatter(lon, lat, c='black', s=40 if i == 0 else 20)  # depot większy
        if i == 0:
            plt.text(lon + 0.02, lat + 0.02, name, fontsize=9, color='black')
        else:
            plt.text(lon + 0.02, lat + 0.02, name, fontsize=7, color='gray')

    # Rysowanie tras
    for truck_idx, route in enumerate(solution):
        c = colors[truck_idx % len(colors)]
        # Budujemy listę punktów (Kraków + route + Kraków)
        path = [0] + route + [0]
        x_coords = [cities[i][1][1] for i in path]  # lon
        y_coords = [cities[i][1][0] for i in path]  # lat
        plt.plot(x_coords, y_coords, color=c, linewidth=1.5, alpha=0.8, label=f"Truck {truck_idx + 1}")

    plt.title("Tabu Search - VRP Solution")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------------------------
# 10. Uruchomienie algorytmu i prezentacja wyników
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    solution, dist_val = tabu_search(
        max_iters=500,  # Możesz zmienić
        tabu_tenure=8,  # Możesz zmienić
        neighbor_size=100  # Możesz zmienić
    )

    print("\n=== Najlepsze rozwiązanie (Tabu Search z 2-opt) ===")
    for truck_idx, route in enumerate(solution):
        route_names = [cities[r][0] for r in route]
        load = sum(cities[r][2] for r in route)
        print(f"Samochód {truck_idx + 1}: {route_names}, ładunek={load}")
    print(f"\nCałkowity dystans: {dist_val:.2f} km")

    # Wizualizacja
    visualize_solution(solution)