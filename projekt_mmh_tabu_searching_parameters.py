import os
import random
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import openrouteservice

# ------------------------------------------------------------------------------
#  1. Ładowanie klucza API openrouteservice z pliku .env
# ------------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Brak klucza API w .env (API_KEY)! Upewnij się, że plik istnieje.")

# ------------------------------------------------------------------------------
#  2. Dane problemu (5 pojazdów, 30 klientów, depo = Kraków)
# ------------------------------------------------------------------------------
NUM_TRUCKS = 5
TRUCK_CAPACITY = 1000

cities = [
    ("Kraków", (50.06143, 19.93658), 0),  # index 0, depo
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

NUM_CITIES = len(cities)

# ------------------------------------------------------------------------------
#  3. Pobranie macierzy odległości po długości tras z OpenRouteService
# ------------------------------------------------------------------------------
client = openrouteservice.Client(key=API_KEY)

coords = [(lon, lat) for (_, (lat, lon), _) in cities]
resp = client.distance_matrix(
    locations=coords,
    profile='driving-car', # według jechania autem z miejsca do miejsca (jak nawigacja)
    metrics=['distance'], # dystans zamiast czasu
    validate=False
)

dist_matrix_meters = resp['distances']
distance_matrix = {}
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        if i == j:
            distance_matrix[(i, j)] = 0.0
        else:
            distance_matrix[(i, j)] = dist_matrix_meters[i][j] / 1000.0  # w km


# ------------------------------------------------------------------------------
#  4. Funkcje pomocnicze
# ------------------------------------------------------------------------------
def route_distance(route):
    """Dystans dla jednej trasy, start i koniec w index 0 (Kraków)."""
    if not route:
        return 0.0
    dist = distance_matrix[(0, route[0])]
    for i in range(len(route) - 1):
        dist += distance_matrix[(route[i], route[i + 1])]
    dist += distance_matrix[(route[-1], 0)]
    return dist


def total_distance(solution):
    """Suma dystansów dla wszystkich tras w solution."""
    return sum(route_distance(r) for r in solution)


def is_feasible(solution):
    """Czy w żadnej trasie nie przekroczono ładowności 1000?"""
    for route in solution:
        load = sum(cities[idx][2] for idx in route)
        if load > TRUCK_CAPACITY:
            return False
    return True


def generate_initial_solution():
    """Rozrzucamy klientów 1..30 w sposób losowy na 5 pojazdów."""
    clients = list(range(1, NUM_CITIES))
    random.shuffle(clients)
    solution = [[] for _ in range(NUM_TRUCKS)]
    for city_idx in clients:
        placed = False
        for t in range(NUM_TRUCKS):
            cur_load = sum(cities[c][2] for c in solution[t])
            if cur_load + cities[city_idx][2] <= TRUCK_CAPACITY:
                solution[t].append(city_idx)
                placed = True
                break
        if not placed:
            # awaryjnie dodaj do ostatniego
            solution[-1].append(city_idx)
    return solution


# ------------------------------------------------------------------------------
#  5. Lokalna optymalizacja (2-opt) w obrębie pojedynczej trasy
# ------------------------------------------------------------------------------
def two_opt(route):
    """Wykonuje 2-opt na liście klientów w pojedynczej trasie."""
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
    """2-opt dla każdej trasy w solution."""
    improved_sol = []
    for route in solution:
        new_r = two_opt(route)
        improved_sol.append(new_r)
    return improved_sol


# ------------------------------------------------------------------------------
#  6. Sąsiedztwo: operator swap, move, swap_within_route
# ------------------------------------------------------------------------------
def swap_cities(solution):
    """Zamiana dwóch klientów między różnymi pojazdami."""
    new_sol = [r[:] for r in solution]
    t1, t2 = random.sample(range(NUM_TRUCKS), 2)
    if not new_sol[t1] or not new_sol[t2]:
        return new_sol
    i1 = random.randint(0, len(new_sol[t1]) - 1)
    i2 = random.randint(0, len(new_sol[t2]) - 1)
    new_sol[t1][i1], new_sol[t2][i2] = new_sol[t2][i2], new_sol[t1][i1]
    return new_sol


def move_city(solution):
    """Przeniesienie jednego klienta z trasy t1 do t2."""
    new_sol = [r[:] for r in solution]
    non_empty = [k for k in range(NUM_TRUCKS) if new_sol[k]]
    if not non_empty:
        return new_sol
    t1 = random.choice(non_empty)
    t2 = random.choice(range(NUM_TRUCKS))
    if t1 == t2:
        return new_sol
    i1 = random.randint(0, len(new_sol[t1]) - 1)
    cityA = new_sol[t1].pop(i1)
    new_sol[t2].append(cityA)
    return new_sol


def swap_within_route(solution):
    """Zamiana dwóch klientów w obrębie jednej trasy."""
    new_sol = [r[:] for r in solution]
    candidates = [i for i in range(NUM_TRUCKS) if len(new_sol[i]) > 1]
    if not candidates:
        return new_sol
    t = random.choice(candidates)
    if len(new_sol[t]) < 2:
        return new_sol
    i1, i2 = random.sample(range(len(new_sol[t])), 2)
    new_sol[t][i1], new_sol[t][i2] = new_sol[t][i2], new_sol[t][i1]
    return new_sol


def generate_neighbors(solution, k):
    """Generuje k sąsiadów, każdy ulepszany 2-optem."""
    neighbors = []
    ops = [swap_cities, move_city, swap_within_route]
    for _ in range(k):
        op = random.choice(ops)
        nsol = op(solution)
        nsol_ls = local_search_2opt(nsol)
        neighbors.append(nsol_ls)
    return neighbors


# ------------------------------------------------------------------------------
#  7. Tabu Search
# ------------------------------------------------------------------------------
def tabu_search(maxTabuSize, neighborhood_size, stoppingTurn):
    """
    Używamy listy tabu o maksymalnym rozmiarze 'maxTabuSize'.
    Generujemy 'neighborhood_size' sąsiadów w każdej iteracji.
    Zatrzymujemy się, jeśli przez 'stoppingTurn' kolejnych iteracji nie udało się poprawić best_solution.
    """
    current_solution = generate_initial_solution()
    while not is_feasible(current_solution):
        current_solution = generate_initial_solution()

    best_solution = current_solution
    best_cost = total_distance(best_solution)

    # Lista rozwiązań tabu
    tabu_list = [best_solution]

    no_improve_turn = 0
    iteration = 0

    while True:
        iteration += 1
        neighbors = generate_neighbors(current_solution, k=neighborhood_size)

        best_candidate = None
        best_candidate_cost = float('inf')

        for cand in neighbors:
            if not is_feasible(cand):
                continue
            cand_cost = total_distance(cand)

            # Sprawdzamy, czy kandydat jest na liście tabu
            if cand in tabu_list and cand_cost >= best_cost:
                # pominąć (chyba że aspiracja -> cand_cost < best_cost)
                continue

            # jeśli jest lepszy od dotychczasowego best_candidate - aktualizujemy
            if cand_cost < best_candidate_cost:
                best_candidate = cand
                best_candidate_cost = cand_cost

        if best_candidate is None:
            # Brak feasible kandydata spoza tabu -> przerwanie
            print("Brak kandydata spoza tabu. Koniec.")
            break

        # Aktualizujemy bieżące rozwiązanie
        current_solution = best_candidate
        current_cost = best_candidate_cost

        # Czy poprawiliśmy best?
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost
            no_improve_turn = 0
        else:
            no_improve_turn += 1

        # Dodajemy nowego kandydata do tabu_list
        tabu_list.append(best_candidate)
        if len(tabu_list) > maxTabuSize:
            # usuwamy najstarsze (fifo)
            tabu_list.pop(0)

        # Kryterium stopu: zbyt wiele iteracji bez poprawy
        if no_improve_turn >= stoppingTurn:
            print(f"Osiągnięto {stoppingTurn} tur bez poprawy. Koniec.")
            break

        # Podgląd co 10 iteracji
        if iteration % 10 == 0:
            print(f"Iteracja {iteration}, best_cost={best_cost:.2f}, no_improve_turn={no_improve_turn}")

    return best_solution, best_cost


# ------------------------------------------------------------------------------
#  8. Wizualizacja wyników
# ------------------------------------------------------------------------------
def visualize_solution(solution):
    colors = ["red", "green", "blue", "orange", "purple"]
    plt.figure(figsize=(8, 8))
    # Rysuj miasta
    for i, (name, (lat, lon), _) in enumerate(cities):
        plt.scatter(lon, lat, c='black' if i != 0 else 'yellow', s=60 if i == 0 else 20)
        plt.text(lon + 0.02, lat + 0.02, name, fontsize=7)
    # Rysuj trasy
    for t_idx, route in enumerate(solution):
        path = [0] + route + [0]
        xx = [cities[i][1][1] for i in path]  # lon
        yy = [cities[i][1][0] for i in path]  # lat
        c = colors[t_idx % len(colors)]
        plt.plot(xx, yy, color=c, label=f"Truck {t_idx + 1}")
    plt.title("Tabu Search")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    plt.show()


# ------------------------------------------------------------------------------
#  9. Uruchomienie z różnymi kombinacjami parametrów w celu znalezienia najlepszego
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Przykładowe zakresy wartości dla testów
    maxTabuSize_values = [20, 30, 40, 50, 60]
    neighborhood_size_values = [20, 30, 40, 50, 60]
    stoppingTurn_values = [50, 60, 70, 80, 90]

    best_overall = None
    best_dist_overall = float('inf')
    best_params = (None, None, None)

    # Testujemy wszystkie kombinacje
    for msize in maxTabuSize_values:
        for nsize in neighborhood_size_values:
            for sturn in stoppingTurn_values:
                print("\n---------------------------------")
                print(f"Test params: maxTabuSize={msize}, neighborhood_size={nsize}, stoppingTurn={sturn}")

                solution, dist_val = tabu_search(msize, nsize, sturn)
                print(f"-> Zakończono dla parametrów (msize={msize}, nsize={nsize}, sturn={sturn})")
                print(f"-> Uzyskany dystans = {dist_val:.2f}")

                if dist_val < best_dist_overall:
                    best_dist_overall = dist_val
                    best_overall = solution
                    best_params = (msize, nsize, sturn)

    # Po zakończeniu pętli wszystkie kombinacje przetestowane
    print("\n=== Najlepszy wynik ze wszystkich kombinacji ===")
    print(f"Parametry: maxTabuSize={best_params[0]}, neighborhood_size={best_params[1]}, stoppingTurn={best_params[2]}")
    print(f"Dystans: {best_dist_overall:.2f}")
    for i, route in enumerate(best_overall):
        names = [cities[r][0] for r in route]
        load = sum(cities[r][2] for r in route)
        print(f"Samochód {i + 1}: {names}, ładunek={load}")

    # Opcjonalnie wizualizacja najlepszej
    visualize_solution(best_overall)