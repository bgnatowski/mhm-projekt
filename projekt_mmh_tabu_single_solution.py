import os
import random
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import openrouteservice
import pandas as pd

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
NUM_TRUCKS = 5 # Liczba dostępnych pojazdów.
TRUCK_CAPACITY = 1000 # Maksymalna pojemność jednego pojazdu (ładowność).

# Lista miast: (nazwa, współrzędne geograficzne (szerokość, długość), zapotrzebowanie na ładunek).
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

NUM_CITIES = len(cities) # Liczba miast w problemie (włącznie z depo).

# ------------------------------------------------------------------------------
#  3. Pobranie macierzy odległości po długości tras z OpenRouteService
# ------------------------------------------------------------------------------
client = openrouteservice.Client(key=API_KEY)
# Współrzędne miast w formacie (longitude, latitude).
coords = [(lon, lat) for (_, (lat, lon), _) in cities]
# Wywołanie API, aby pobrać macierz odległości pomiędzy miastami.
resp = client.distance_matrix(
    locations=coords,       # Lista współrzędnych wszystkich miast.
    profile='driving-car',  # Tryb: jazda samochodem.
    metrics=['distance'],   # Interesuje nas tylko dystans, nie czas.
    validate=False          # Wyłączenie walidacji danych wejściowych (przyśpiesza zapytanie).
)

# Ekstrakcja surowej macierzy odległości w metrach z odpowiedzi API.
dist_matrix_meters = resp['distances']
# Przetwarzanie macierzy odległości do czytelnego słownika.
distance_matrix = {}
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        if i == j:
            distance_matrix[(i, j)] = 0.0 # Jeśli miasto to samo, dystans wynosi 0 (brak podróży).
        else:
            distance_matrix[(i, j)] = dist_matrix_meters[i][j] / 1000.0  # Konwersja odległości z metrów na kilometry.

# ------------------------------------------------------------------------------
#  4. Funkcje pomocnicze
# ------------------------------------------------------------------------------
def save_distance_matrix_to_csv(distance_matrix, cities, filename="distance_matrix.csv"):
    """
    Zapisuje macierz odległości do pliku CSV z faktycznymi nazwami miast.

    Args:
        distance_matrix (dict): Słownik z odległościami {(i, j): dystans}.
        cities (list): Lista miast z ich nazwami i współrzędnymi.
        filename (str): Nazwa pliku wyjściowego.
    """
    # Liczba miast (zakładamy, że macierz jest kwadratowa)
    num_cities = int(len(distance_matrix) ** 0.5)

    # Wyciągamy nazwy miast z listy cities
    city_names = [city[0] for city in cities]

    # Tworzymy DataFrame na podstawie distance_matrix
    data = []
    for i in range(num_cities):
        row = [distance_matrix[(i, j)] for j in range(num_cities)]
        data.append(row)

    df = pd.DataFrame(data, index=city_names, columns=city_names)

    # Zapisujemy DataFrame do pliku CSV
    df.to_csv(filename, index=True)

    print(f"Macierz odległości została zapisana do pliku {filename}.")

def route_distance(route):
    """Dystans dla jednej trasy, start i koniec w index 0 (Kraków)."""
    if not route:
        return 0.0
    #  Dystans od bazy (miasto 0) do pierwszego klienta w trasie.
    dist = distance_matrix[(0, route[0])]

    # Dodajemy dystanse pomiędzy kolejnymi miastami w trasie.
    for i in range(len(route) - 1):
        dist += distance_matrix[(route[i], route[i + 1])]

    # Dodajemy dystans z ostatniego miasta w trasie z powrotem do bazy.
    dist += distance_matrix[(route[-1], 0)]
    return dist


def total_distance(solution):
    """Suma dystansów dla wszystkich tras w solution."""
    return sum(route_distance(r) for r in solution)


def is_feasible(solution):
    """Czy w żadnej trasie nie przekroczono ładowności 1000?"""
    for route in solution:
        load = sum(cities[idx][2] for idx in route)  # Obliczamy całkowity ładunek w trasie
        if load > TRUCK_CAPACITY:
            return False
    return True


def generate_initial_solution():
    """Rozrzucamy klientów 1..30 w sposób losowy na 5 pojazdów."""
    clients = list(range(1, NUM_CITIES))
    random.shuffle(clients)
    solution = [[] for _ in range(NUM_TRUCKS)] # lista list, gdzie każda lista reprezentuje trasy jednego pojazdu
    # Iterujemy przez przetasowaną listę klientów.
    for city_idx in clients:
        placed = False # Flaga wskazująca, czy klient został przypisany do jakiegoś pojazdu.

        # Próba przypisania klienta do każdego z pojazdów.
        for t in range(NUM_TRUCKS):
            # Obliczamy bieżące obciążenie pojazdu (suma zapotrzebowania klientów w trasie).
            cur_load = sum(cities[c][2] for c in solution[t])

            # Sprawdzamy, czy dodanie nowego klienta nie przekroczy pojemności pojazdu.
            if cur_load + cities[city_idx][2] <= TRUCK_CAPACITY:
                solution[t].append(city_idx)
                placed = True
                break
        # Jeśli klient nie mógł zostać przypisany do żadnego pojazdu:
        if not placed:
            # awaryjnie dodaj do ostatniego pojazdu
            solution[-1].append(city_idx)
    return solution


# ------------------------------------------------------------------------------
#  5. Lokalna optymalizacja (2-opt) w obrębie pojedynczej trasy
# ------------------------------------------------------------------------------
def two_opt(route):
    """Wykonuje 2-opt na liście klientów w pojedynczej trasie."""
    if len(route) < 2: # Jeśli trasa zawiera mniej niż 2 miasta, nic nie optymalizujemy.
        return route

    # Kopiujemy początkową trasę i obliczamy jej dystans.
    best_route = route[:]
    best_dist = route_distance(best_route)
    improved = True # Flaga wskazująca, czy znaleziono lepsze rozwiązanie.

    while improved:
        improved = False
        # Iterujemy przez wszystkie możliwe pary miast w trasie.
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                # Tworzymy nową trasę, odwracając fragment między miastami i, j.
                new_route = best_route[:]
                new_route[i:j + 1] = reversed(new_route[i:j + 1])

                # Obliczamy dystans nowej trasy.
                new_dist = route_distance(new_route)

                # Jeśli nowa trasa ma mniejszy dystans, aktualizujemy najlepszą trasę.
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
    # Tworzymy nową listę tras po lokalnej optymalizacji.
    improved_sol = []

    # Dla każdej trasy w rozwiązaniu stosujemy 2-opt.
    for route in solution:
        new_r = two_opt(route)
        improved_sol.append(new_r)
    return improved_sol


# ------------------------------------------------------------------------------
#  6. Sąsiedztwo: operator swap, move, swap_within_route
# ------------------------------------------------------------------------------
def swap_cities(solution):
    """Zamiana dwóch klientów między różnymi pojazdami."""
    # Tworzymy kopię rozwiązania, aby uniknąć modyfikacji oryginału
    new_sol = [r[:] for r in solution]
    # Wybieramy losowo dwa różne pojazdy.
    t1, t2 = random.sample(range(NUM_TRUCKS), 2)
    # Jeśli którykolwiek z pojazdów ma pustą trasę, nie wykonujemy zamiany.
    if not new_sol[t1] or not new_sol[t2]:
        return new_sol
    # Wybieramy losowo jednego klienta z każdej trasy.
    i1 = random.randint(0, len(new_sol[t1]) - 1)
    i2 = random.randint(0, len(new_sol[t2]) - 1)
    # Wymieniamy klientów między trasami.
    new_sol[t1][i1], new_sol[t2][i2] = new_sol[t2][i2], new_sol[t1][i1]
    return new_sol


def move_city(solution):
    """Przeniesienie jednego klienta z trasy t1 do t2."""
    # Tworzymy kopię rozwiązania
    new_sol = [r[:] for r in solution]
    # Wybieramy tylko te pojazdy, które mają przynajmniej jednego klienta
    non_empty = [k for k in range(NUM_TRUCKS) if new_sol[k]]
    # Jeśli wszystkie trasy są puste, zwracamy rozwiązanie bez zmian.
    if not non_empty:
        return new_sol
    # Wybieramy losowo trasę (t1) i docelową trasę (t2).
    t1 = random.choice(non_empty)
    t2 = random.choice(range(NUM_TRUCKS))
    # Jeśli wybraliśmy tę samą trasę, nic nie zmieniamy.
    if t1 == t2:
        return new_sol
    # Wybieramy losowego klienta z trasy t1.
    i1 = random.randint(0, len(new_sol[t1]) - 1)
    cityA = new_sol[t1].pop(i1) # Usuwamy klienta z t1.
    new_sol[t2].append(cityA) # Dodajemy klienta do trasy t2.
    return new_sol


def swap_within_route(solution):
    """Zamiana dwóch klientów w obrębie jednej trasy."""
    new_sol = [r[:] for r in solution]
    # Wybieramy pojazdy, które mają co najmniej dwóch klientów.
    candidates = [i for i in range(NUM_TRUCKS) if len(new_sol[i]) > 1]
    # Jeśli żaden pojazd nie spełnia warunku, zwracamy rozwiązanie bez zmian.
    if not candidates:
        return new_sol
    # Wybieramy losowo jeden pojazd.
    t = random.choice(candidates)
    # Jeśli trasa wybranego pojazdu ma mniej niż dwóch klientów, nic nie zmieniamy.
    if len(new_sol[t]) < 2:
        return new_sol
    # Wybieramy losowo dwóch klientów w trasie.
    i1, i2 = random.sample(range(len(new_sol[t])), 2)
    new_sol[t][i1], new_sol[t][i2] = new_sol[t][i2], new_sol[t][i1] # Wymieniamy ich miejscami.
    return new_sol


def generate_neighbors(solution, k):
    """Generuje k sąsiadów, każdy ulepszany 2-optem."""
    neighbors = []
    # Lista operatorów generujących sąsiadów.
    ops = [swap_cities, move_city, swap_within_route]
    for _ in range(k):
        op = random.choice(ops)  # Wybieramy losowy operator.
        nsol = op(solution) # Tworzymy nowego sąsiada, stosując operator do bieżącego rozwiązania.
        nsol_ls = local_search_2opt(nsol)  # Ulepszamy trasę sąsiada za pomocą 2-opt.
        neighbors.append(nsol_ls) # Dodajemy ulepszonego sąsiada do listy.
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
    # Jeśli na jakiejś trasie przekroczono ładownosc to wygeneruj ponownie
    while not is_feasible(current_solution):
        current_solution = generate_initial_solution()

    best_solution = current_solution
    best_cost = total_distance(best_solution) # Obliczamy koszt dla najlepszego rozwiązania.

    # Lista rozwiązań tabu
    tabu_list = [best_solution]

    no_improve_turn = 0  # Licznik iteracji bez poprawy najlepszego rozwiązania.
    iteration = 0

    while True:
        iteration += 1
        # Generujemy sąsiadów bieżącego rozwiązania.
        neighbors = generate_neighbors(current_solution, k=neighborhood_size)

        # Inicjalizujemy najlepszego kandydata z bardzo wysokim kosztem (INF).
        best_candidate = None
        best_candidate_cost = float('inf')

        # Iterujemy przez sąsiadów, aby znaleźć najlepszego kandydata.
        for cand in neighbors:
            if not is_feasible(cand): # Jeśli rozwiązanie nie spełnia ograniczeń, pomijamy je.
                continue
            cand_cost = total_distance(cand) # Jeśli spełnia obliczamy koszt

            # Sprawdzamy, czy kandydat jest na liście tabu
            if cand in tabu_list and cand_cost >= best_cost:
                # Jeśli jest na liście tabu i nie spełnia kryterium aspiracji (lepszy niż best_cost), pomijamy.
                continue

            # Jeśli jest na liście tabu i nie spełnia kryterium aspiracji (lepszy niż best_cost), pomijamy.
            if cand_cost < best_candidate_cost:
                best_candidate = cand
                best_candidate_cost = cand_cost

        # Jeśli nie znaleziono żadnego kandydata spoza listy tabu, kończymy algorytm.
        if best_candidate is None:
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
            # Jeśli lista tabu przekroczyła maksymalny rozmiar, usuwamy najstarsze rozwiązanie (FIFO).
            tabu_list.pop(0)

        # Kryterium stopu: zbyt wiele iteracji bez poprawy konczymy algorytm
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
#  9. Uruchomienie z parametrami
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Zmiana parametrów
    # msize = 20      # maxTabuSize
    # nsize = 50      # neighborhood_size
    # sturn = 80      # stoppingTurn

    # msize = 50      # maxTabuSize
    # nsize = 50      # neighborhood_size
    # sturn = 100     # stoppingTurn

    msize = 20      # maxTabuSize
    nsize = 50      # neighborhood_size
    sturn = 80      # stoppingTurn

    solution, dist_val = tabu_search(
        maxTabuSize=msize,
        neighborhood_size=nsize,
        stoppingTurn=sturn
    )

    print(f"\n=== Najlepsze rozwiązanie (Tabu Search) ===")
    for idx, route in enumerate(solution):
        names = [cities[r][0] for r in route]
        load = sum(cities[r][2] for r in route)
        print(f"Samochód {idx+1}: {names}, ładunek={load}")
    print(f"Całkowity dystans: {dist_val:.2f} km")

    visualize_solution(solution)
    save_distance_matrix_to_csv(distance_matrix, cities, "distance_matrix.csv")